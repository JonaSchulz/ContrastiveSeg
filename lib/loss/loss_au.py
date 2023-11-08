import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC

from lib.loss.loss_helper import FSAuxCELoss, FSAuxRMILoss, FSCELoss
from lib.utils.tools.logger import Logger as Log


class AULoss(nn.Module, ABC):
    def __init__(self, configer):
        super(AULoss, self).__init__()

        self.configer = configer

        self.alpha = 2
        self.t = 2
        self.loss_weight = 1
        self.method = 1
        if self.configer.exists('loss', 'au_loss'):
            self.alpha = self.configer.get('loss', 'au_loss', 'alpha')
            self.t = self.configer.get('loss', 'au_loss', 't')
            self.loss_weight = self.configer.get('loss', 'au_loss', 'loss_weight')
            self.method = self.configer.get('loss', 'au_loss', 'method')

        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

        self.max_samples = self.configer.get('contrast', 'max_samples')
        self.max_views = self.configer.get('contrast', 'max_views')

    def _hard_anchor_sampling(self, X, y_hat, y):
        """
        return:
        X_: - shape: (num_classes_to_be_sampled_from, num_sample_pixels_per_class, proj_dim)
            - contains all sampled pixel embeddings, separate for each class
        y_: - shape: (num_classes_to_be_sampled_from)
            - maps index of class samples from X_ to class label (X_ is not ordered from lowest to highest class label)
        """

        # X = feats, y_hat = labels, y = predict
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        # pick out classes to be used from ground-truth labels
        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]  # current label image
            this_classes = torch.unique(this_y)     # classes contained in current label image
            this_classes = [x for x in this_classes if x != self.ignore_label]  # remove classes to be ignored
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]   # remove class
            # if there are <= max_views pixels of that class in the current label image
            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            Log.info(f"Not enough samples. Labelmap:\n{y_hat}")
            return None, None

        n_view = self.max_samples // total_classes  # evenly sample from all classes to get a total of max_samples samples
        n_view = min(n_view, self.max_views)    # number of samples per class may not exceed max_views

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()   # X_ shape (total_classes, samples_per_class, proj_dim)
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]  # current ground-truth label
            this_y = y[ii]  # current predicted label
            this_classes = classes[ii]  # classes to be sampled from for current image

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()  # indices of all pixels with
                # ground-truth label cls_id and predicted label not cls_id
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()  # indices of all pixels with
                # ground-truth label cls_id and predicted label cls_id

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                # if possible take half of sample pixels from hard and half from easy pixels
                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                # else if enough hard pixels for half the samples keep as many easy pixels as possible and fill up with
                # hard pixels
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                # else if enough easy pixels for half the samples keep as many hard pixels as possible and fill up with
                # easy pixels
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]   # indices of hard pixels to be used
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]   # indices of easy pixels to be used
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)  # store sampled pixel embeddings in X_
                y_[X_ptr] = cls_id  # store class corresponding to each entry in X_
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        # feats_: shape (num_classes_sampled_from, samples_per_class, proj_dim)
        # labels_: shape (num_classes_sampled_from)

        anchor_num, n_view = feats_.shape[0], feats_.shape[1]   # n_view: number of samples per class,
        # anchor_num: number of classes with samples

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        # alignment:
        align = 0
        for cls in feats_:
            align += (F.pdist(cls) ** self.alpha).mean().item()
        align /= anchor_num

        # uniformity:

        # method 1 (sq_dist for every anchor and all negatives for that anchor)
        if self.method == 1:
            feats_flat = torch.transpose(feats_, 0, 1).flatten(end_dim=-2)
            sq_dist = torch.norm(feats_flat.unsqueeze(dim=0).transpose(0, 1) - feats_flat, dim=-1) ** 2
            labels_ = labels_.contiguous().view(-1, 1)
            mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).cuda()
            mask = ~mask.repeat(contrast_count, contrast_count)
            sq_dist_masked = torch.masked_select(sq_dist, mask)

            uniform = torch.exp(-self.t * sq_dist_masked)
            uniform = torch.log(uniform.mean())

        # method 2 (pairwise sq_dist between pairwise negative groups of samples):
        elif self.method == 2:
            feats_t = torch.transpose(feats_, 0, 1)
            num_sets = feats_t.size(0)
            uniform = 0
            for set in feats_t:
                sq_dist = torch.pdist(set) ** 2
                # uniform += torch.log(torch.exp(-self.t * sq_dist).mean())
                uniform += torch.exp(-self.t * sq_dist).mean()
            uniform /= num_sets
            uniform = torch.log(uniform)

        # sq_dist = torch.cdist(feats_t, feats_t) ** 2
        # sq_dist_dim = sq_dist.size(1)
        # sq_dist = sq_dist[:, torch.tril_indices(sq_dist_dim, sq_dist_dim).unbind()]
        # uniform = torch.exp(-self.t * sq_dist)
        # uniform = uniform.sum(1)
        # uniform = torch.log(uniform.mean())

        return align + self.loss_weight * uniform

    def forward(self, feats, labels=None, predict=None):
        # feats: output of proj_head
        # labels: ground-truth labels
        # predict: predicted labels from seg_head
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)   # labels shape (batch_size, h * w)
        predict = predict.contiguous().view(batch_size, -1)     # predict shape (batch_size, h * w)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])    # feats shape (batch_size, h * w, proj_dim)

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)
        return loss


class AUCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(AUCELoss, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.loss_weight = self.configer.get('contrast', 'loss_weight')
        self.use_rmi = self.configer.get('contrast', 'use_rmi')

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSCELoss(configer=configer)

        self.contrast_criterion = AULoss(configer=configer)

        # Tensorboard:
        self.loss = 0
        self.loss_contrast = 0

    def forward(self, preds, target, with_embed=False):
        h, w = target.size(1), target.size(2)

        assert "seg" in preds
        assert "embed" in preds

        seg = preds['seg']
        embedding = preds['embed']

        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)

        _, predict = torch.max(seg, 1)
        loss_contrast = self.contrast_criterion(embedding, target, predict)

        # Tensorboard:
        self.loss = loss
        self.loss_contrast = loss_contrast

        if with_embed is True:
            return loss + self.loss_weight * loss_contrast

        return loss + 0 * loss_contrast  # just a trick to avoid errors in distributed training



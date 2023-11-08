from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.loss.loss_helper import FSAuxCELoss, FSAuxRMILoss, FSCELoss
from lib.utils.tools.logger import Logger as Log


class PixelContrastLoss(nn.Module, ABC):
    def __init__(self, configer):
        super(PixelContrastLoss, self).__init__()

        self.configer = configer
        self.temperature = self.configer.get('contrast', 'temperature')
        self.base_temperature = self.configer.get('contrast', 'base_temperature')

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

        labels_ = labels_.contiguous().view(-1, 1)  # labels_ shape (num_classes_sampled_from, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

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


class SingleImagePixelContrastLoss(nn.Module, ABC):
    def __init__(self, configer):
        super(SingleImagePixelContrastLoss, self).__init__()

        self.configer = configer
        self.temperature = self.configer.get('contrast', 'temperature')
        self.base_temperature = self.configer.get('contrast', 'base_temperature')

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
        total_classes_per_image = []
        for ii in range(batch_size):
            this_y = y_hat[ii]  # current label image
            this_classes = torch.unique(this_y)     # classes contained in current label image
            this_classes = [x for x in this_classes if x != self.ignore_label]  # remove classes to be ignored
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]   # remove class
            # if there are <= max_views pixels of that class in the current label image
            classes.append(this_classes)
            total_classes += len(this_classes)
            total_classes_per_image.append(len(this_classes))

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes  # evenly sample from all classes to get a total of max_samples samples
        n_view = min(n_view, self.max_views)    # number of samples per class may not exceed max_views

        X_imagewise = []
        y_imagewise = []

        for ii in range(batch_size):
            X_ = torch.zeros((total_classes_per_image[ii], n_view, feat_dim),
                             dtype=torch.float).cuda()  # X_ shape (total_classes, samples_per_class, proj_dim)
            y_ = torch.zeros(total_classes_per_image[ii], dtype=torch.float).cuda()

            this_y_hat = y_hat[ii]  # current ground-truth label
            this_y = y[ii]  # current predicted label
            this_classes = classes[ii]  # classes to be sampled from for current image

            for cls_idx, cls_id in enumerate(this_classes):
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

                X_[cls_idx, :, :] = X[ii, indices, :].squeeze(1)  # store sampled pixel embeddings in X_
                y_[cls_idx] = cls_id  # store class corresponding to each entry in X_

                X_imagewise.append(X_)
                y_imagewise.append(y_)

        return X_imagewise, y_imagewise

    def _contrastive(self, feats_, labels_):
        # feats_: shape (num_classes_sampled_from, samples_per_class, proj_dim)
        # labels_: shape (num_classes_sampled_from)

        anchor_num, n_view = feats_.shape[0], feats_.shape[1]   # n_view: number of samples per class,
        # anchor_num: number of classes with samples

        labels_ = labels_.contiguous().view(-1, 1)  # labels_ shape (num_classes_sampled_from, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

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

        loss = 0
        for feat, label in zip(feats_, labels_):
            loss += self._contrastive(feat, label)
        loss /= batch_size

        return loss


class ContrastCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(ContrastCELoss, self).__init__()

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

        self.contrast_criterion = PixelContrastLoss(configer=configer)

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


class SingleImageContrastCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(SingleImageContrastCELoss, self).__init__()

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

        self.contrast_criterion = SingleImagePixelContrastLoss(configer=configer)

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


class ContrastAuxCELoss(nn.Module, ABC):
    def __init__(self, configer=None):
        super(ContrastAuxCELoss, self).__init__()

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
            self.seg_criterion = FSAuxCELoss(configer=configer)

        self.contrast_criterion = PixelContrastLoss(configer=configer)

    def forward(self, preds, target, with_embed=False):
        h, w = target.size(1), target.size(2)

        assert "seg" in preds
        assert "seg_aux" in preds
        assert "embed" in preds

        seg = preds['seg']
        seg_aux = preds['seg_aux']
        embedding = preds['embed']

        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        pred_aux = F.interpolate(input=seg_aux, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion([pred_aux, pred], target)

        _, predict = torch.max(seg, 1)
        loss_contrast = self.contrast_criterion(embedding, target, predict)

        if with_embed is True:
            return loss + self.loss_weight * loss_contrast

        return loss + 0 * loss_contrast  # just a trick to avoid errors in distributed training


##################################
# Sequential Architecture:

class PixelContrastLossSeq(nn.Module, ABC):
    def __init__(self, configer):
        super(PixelContrastLossSeq, self).__init__()

        self.configer = configer
        self.temperature = self.configer.get('contrast', 'temperature')
        self.base_temperature = self.configer.get('contrast', 'base_temperature')

        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

        self.max_samples = self.configer.get('contrast', 'max_samples')
        self.max_views = self.configer.get('contrast', 'max_views')

    def _hardest_example_sampling(self, X, y_hat, X_anchor, y_anchor):
        """
        Args:
            X: input image batch (batch_size, height * width, proj_dim)
            y_hat: input target label batch (batch_size, height * width)
            X_anchor: (num_classes, samples_per_class, proj_dim)
            y_anchor: mapping of class indices to class labels in X_anchor (num_classes)

        Returns:

        """

        batch_size, n_view, feat_dim = X_anchor.shape[0], X_anchor.shape[1], X_anchor.shape[2]
        classes = y_anchor
        total_classes = y_anchor.size

        # number of positive/negative samples per class:
        num_positives = ...
        num_negatives = ...

        # remove batch dimension in input data:
        X = X.view(-1, 3)
        y_hat = y_hat.view(-1)

        if total_classes == 0:
            return None, None

        X_ = torch.zeros((total_classes, n_view, feat_dim),
                         dtype=torch.float).cuda()  # X_ shape (total_classes, samples_per_class, proj_dim)
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        for cls_idx, cls_id in enumerate(classes):
            this_X_anchor = X_anchor[cls_idx]
            this_pos_indices = (y_hat == cls_id)
            this_neg_indices = ~this_pos_indices
            this_positives = X[this_pos_indices]
            this_negatives = X[this_neg_indices]
            anchor_dot_pos = torch.matmul(this_X_anchor, torch.transpose(this_positives, 0, 1))
            anchor_dot_neg = torch.matmul(this_X_anchor, torch.transpose(this_negatives, 0, 1))
            hardest_pos_indices = torch.topk(-anchor_dot_pos, num_positives, 1).indices
            hardest_neg_indices = torch.topk(anchor_dot_pos, num_negatives, 1).indices
            hardest_positives = this_positives[[(i, j) for i, j in enumerate(hardest_pos_indices)]]

    def _random_anchor_sampling(self, X, y_hat):
        # X = feats, y_hat = labels, y = predict
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        # pick out classes to be used from ground-truth labels
        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]  # current label image
            this_classes = torch.unique(this_y)  # classes contained in current label image
            this_classes = [x for x in this_classes if x != self.ignore_label]  # remove classes to be ignored
            this_classes = [x for x in this_classes if
                            (this_y == x).nonzero().shape[0] > self.max_views]  # remove class
            # if there are <= max_views pixels of that class in the current label image
            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            Log.info(f"Not enough samples. Labelmap:\n{y_hat}")
            return None, None

        n_view = self.max_samples // total_classes  # evenly sample from all classes to get a total of max_samples samples
        n_view = min(n_view, self.max_views)  # number of samples per class may not exceed max_views

        X_ = torch.zeros((total_classes, n_view, feat_dim),
                         dtype=torch.float).cuda()  # X_ shape (total_classes, samples_per_class, proj_dim)
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]  # current ground-truth label
            this_classes = classes[ii]  # classes to be sampled from for current image

            for cls_id in this_classes:
                indices = (this_y_hat == cls_id).nonzero()
                num = indices.shape[0]
                perm = torch.randperm(num)
                indices = indices[perm[:n_view]]

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)  # store sampled pixel embeddings in X_
                y_[X_ptr] = cls_id  # store class corresponding to each entry in X_
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        # feats_: shape (num_classes_sampled_from, samples_per_class, proj_dim)
        # labels_: shape (num_classes_sampled_from)

        anchor_num, n_view = feats_.shape[0], feats_.shape[1]   # n_view: number of samples per class,
        # anchor_num: number of classes with samples

        labels_ = labels_.contiguous().view(-1, 1)  # labels_ shape (num_classes_sampled_from, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None):
        # feats: output of proj_head
        # labels: ground-truth labels
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)   # labels shape (batch_size, h * w)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])    # feats shape (batch_size, h * w, proj_dim)

        feats_, labels_ = self._random_anchor_sampling(feats, labels)

        loss = self._contrastive(feats_, labels_)
        return loss


class ContrastCELossSeq(nn.Module, ABC):
    def __init__(self, configer=None):
        super(ContrastCELossSeq, self).__init__()

        self.configer = configer

        ignore_index = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']
        Log.info('ignore_index: {}'.format(ignore_index))

        self.use_rmi = self.configer.get('contrast', 'use_rmi')

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSCELoss(configer=configer)

        self.contrast_criterion = PixelContrastLossSeq(configer=configer)

        # Tensorboard:
        self.loss = 0

    def forward(self, preds, target, with_embed=False):
        h, w = target.size(1), target.size(2)

        if "seg" in preds:
            seg = preds['seg']
            pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
            loss = self.seg_criterion(pred, target)

        elif "embed" in preds:
            embedding = preds['embed']
            loss = self.contrast_criterion(embedding, target)

        self.loss = loss
        return loss

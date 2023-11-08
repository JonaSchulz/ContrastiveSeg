import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC

from lib.loss.loss_helper import FSAuxCELoss, FSAuxRMILoss, FSCELoss


class DiscrimLoss(nn.Module, ABC):
    def __init__(self, configer, delta_v=0.5, delta_d=1.5, alpha=1, beta=1, gamma=0.001):
        super(DiscrimLoss, self).__init__()

        self.configer = configer
        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

        if self.configer.exists('loss', 'discrim_loss'):
            delta_v = self.configer.get('loss', 'discrim_loss', 'delta_v')
            delta_d = self.configer.get('loss', 'discrim_loss', 'delta_d')
            alpha = self.configer.get('loss', 'discrim_loss', 'alpha')
            beta = self.configer.get('loss', 'discrim_loss', 'beta')
            gamma = self.configer.get('loss', 'discrim_loss', 'gamma')

        self.max_views = self.configer.get('contrast', 'max_views')
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, feats, labels=None):
        """
        Args:
            feats: pixel embeddings (batch_size, height, width, proj_dim)
            labels: (batch_size, height, width)

        Returns:
            alpha * loss_var + beta * loss_dist + gamma * loss_reg
        """
        print(feats.shape)
        print(labels)

        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        feats = feats.permute(0, 2, 3, 1)

        feat_dim = feats.shape[-1]
        feats = feats.contiguous().view(-1, feat_dim)
        labels = labels.view(-1)

        classes = torch.unique(labels)
        classes = [x for x in classes if x != self.ignore_label]
        classes = [x for x in classes if (labels == x).nonzero().shape[0] > self.max_views]
        total_classes = len(classes)

        loss_var = 0
        loss_dist = 0
        loss_reg = 0

        cluster_feats_lst = []
        cluster_mean_lst = []

        for cls_id in classes:
            cluster_feats = feats[labels == cls_id]
            cluster_mean = torch.mean(cluster_feats, dim=0)
            cluster_n = cluster_feats.shape[0]
            cluster_feats_lst.append(cluster_feats)
            cluster_mean_lst.append(cluster_mean)

            # loss_var:
            dist_delta_v = torch.norm(cluster_mean - cluster_feats, dim=1) - self.delta_v
            loss_var += torch.div(torch.sum(dist_delta_v[dist_delta_v > 0] ** 2), cluster_n)

            # loss_reg:
            loss_reg += torch.norm(cluster_mean)

        # loss_dist:
        for cls_idx_a, _ in enumerate(classes):
            other_classes = [x for i, x in enumerate(classes) if i != cls_idx_a]
            cluster_mean_a = cluster_mean_lst[cls_idx_a]
            for cls_idx_b, _ in enumerate(other_classes):
                cluster_mean_b = cluster_mean_lst[cls_idx_b]
                dist_delta_d = 2 * self.delta_d - torch.norm(cluster_mean_a - cluster_mean_b)
                loss_dist += dist_delta_d[dist_delta_d > 0] ** 2

        loss_var /= total_classes
        loss_dist /= (total_classes * (total_classes - 1))
        loss_reg /= total_classes

        return self.alpha * loss_var + self.beta * loss_dist + self.gamma * loss_reg


class DiscrimCELoss(nn.Module, ABC):
    def __init__(self, configer):
        super(DiscrimCELoss, self).__init__()

        self.configer = configer
        self.loss_weight = self.configer.get('contrast', 'loss_weight')
        self.use_rmi = self.configer.get('contrast', 'use_rmi')

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSCELoss(configer=configer)
        self.discrim_criterion = DiscrimLoss(configer)

        # Tensorboard:
        self.loss = 0
        self.loss_contrast = 0

    def forward(self, preds, target, with_embed=False):
        h, w = target.size(1), target.size(2)

        assert "embed" in preds
        assert "seg" in preds

        seg = preds["seg"]
        embedding = preds['embed']

        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
        discrim_loss = self.discrim_criterion(embedding, target)

        # Tensorboard:
        self.loss = loss
        self.loss_contrast = discrim_loss

        if with_embed is True:
            return loss + self.loss_weight * discrim_loss

        return loss + 0 * discrim_loss  # just a trick to avoid errors in distributed training


class DiscrimCELossSeq(nn.Module, ABC):
    def __init__(self, configer):
        super(DiscrimCELossSeq, self).__init__()

        self.configer = configer

        self.use_rmi = self.configer.get('contrast', 'use_rmi')

        if self.use_rmi:
            self.seg_criterion = FSAuxRMILoss(configer=configer)
        else:
            self.seg_criterion = FSCELoss(configer=configer)
        self.discrim_criterion = DiscrimLoss(configer)

        # Tensorboard:
        self.loss = 0

    def forward(self, preds, target, with_embed=False):
        h, w = target.size(1), target.size(2)

        if "embed" in preds:
            embedding = preds['embed']
            loss = self.discrim_criterion(embedding, target)

        elif "seg" in preds:
            seg = preds["seg"]
            pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
            loss = self.seg_criterion(pred, target)

        # Tensorboard:
        self.loss = loss

        return loss

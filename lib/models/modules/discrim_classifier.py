import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscrimClassifier(nn.Module):
    def __init__(self, configer, cluster_centers, cls_ids, delta_v):
        super(DiscrimClassifier, self).__init__()
        self.configer = configer
        self.cluster_centers = cluster_centers
        self.cls_ids = cls_ids
        self.num_classes = len(cluster_centers)
        self.delta_v = delta_v

    def forward(self, x):
        if type(x) is dict:
            x = x["embed"]

        with torch.no_grad():
            batch_size, proj_dim, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
            labelmap = torch.zeros(batch_size, h * w)
            x = x.contiguous().permute(0, 2, 3, 1)
            x = x.view(batch_size, -1, proj_dim)

            for cls_idx, cluster_c in enumerate(self.cluster_centers):
                dist = torch.cdist(x, cluster_c.unsqueeze(0))
                mask = (dist <= self.delta_v)
                mask = mask.squeeze()
                labelmap[mask] = self.cls_ids[cls_idx]

            labelmap = labelmap.view(batch_size, h, w)
            labelmap = F.one_hot(labelmap.to(torch.long))

            return labelmap

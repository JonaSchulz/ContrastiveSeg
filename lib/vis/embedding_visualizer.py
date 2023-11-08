import torch
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from umap import UMAP
import numpy as np


class EmbeddingVisualizer:
    def __init__(self, configer):
        self.configer = configer
        self.method = self.configer.get('embedding_visualizer', 'method')
        self.num_samples = 5000
        self.dot_size = 5

        if self.configer.exists('embedding_visualizer', 'num_samples'):
            self.num_samples = self.configer.get('embedding_visualizer', 'num_samples')

        self.plot_dir = self.configer.get('embedding_visualizer', 'plot_dir')
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.ignore_label = -1
        if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
            self.ignore_label = self.configer.get('loss', 'params')['ce_ignore_index']

    def visualize_embeddings(self, embed, label, name="embed"):
        if self.method == 'pca':
            self.pca(embed, label, name=name)
        elif self.method == 'tsne':
            self.tsne(embed, label, name=name)
        elif self.method == 'umap':
            self.umap(embed, label, name=name)

    def pca(self, embed, label, name):
        label = label.unsqueeze(1).float().clone()
        label = torch.nn.functional.interpolate(label,
                                                (embed.shape[2], embed.shape[3]), mode='nearest')
        label = label.squeeze(1).long()
        assert label.shape[-1] == embed.shape[-1], '{} {}'.format(label.shape, embed.shape)

        proj_dim = embed.shape[1]
        embed = embed.permute(0, 2, 3, 1)
        embed = embed.contiguous().view(-1, proj_dim)
        label = label.contiguous().view(-1)

        u, s, v = torch.pca_lowrank(embed, 2)
        embed_2d = torch.matmul(embed, v[:, :2])

        classes = torch.unique(label)
        classes = [x for x in classes if x != self.ignore_label]

        if self.num_samples == "all":
            fig, ax = plt.subplots()

            for cls_idx, cls_id in enumerate(classes):
                this_embed = embed_2d[label == cls_id]
                this_embed = torch.transpose(this_embed, 0, 1).cpu()
                ax.scatter(this_embed[0], this_embed[1], self.dot_size)

        else:
            max_samples_per_class = self.num_samples // len(classes)

            fig, ax = plt.subplots()

            for cls_idx, cls_id in enumerate(classes):
                this_embed = embed_2d[label == cls_id]
                if this_embed.shape[0] >= max_samples_per_class:
                    perm = torch.randperm(this_embed.shape[0])
                    this_embed = this_embed[perm[:max_samples_per_class]]
                this_embed = torch.transpose(this_embed, 0, 1).cpu()
                ax.scatter(this_embed[0], this_embed[1], self.dot_size)

        fig.savefig(os.path.abspath(self.plot_dir) + f"/{name}_{self.configer.get('iters')}.png")

    def tsne(self, embed, label, name):
        label = label.unsqueeze(1).float().clone()
        label = torch.nn.functional.interpolate(label,
                                                 (embed.shape[2], embed.shape[3]), mode='nearest')
        label = label.squeeze(1).long()
        assert label.shape[-1] == embed.shape[-1], '{} {}'.format(label.shape, embed.shape)

        proj_dim = embed.shape[1]
        embed = embed.permute(0, 2, 3, 1)
        embed = embed.contiguous().view(-1, proj_dim)
        label = label.contiguous().view(-1)

        embed_2d = TSNE(learning_rate='auto', init='pca').fit_transform(embed.cpu())
        embed_2d = torch.from_numpy(embed_2d)

        classes = torch.unique(label)
        classes = [x for x in classes if x != self.ignore_label]

        if self.num_samples == "all":
            fig, ax = plt.subplots()

            for cls_idx, cls_id in enumerate(classes):
                this_embed = embed_2d[label == cls_id]
                this_embed = torch.transpose(this_embed, 0, 1).cpu()
                ax.scatter(this_embed[0], this_embed[1], self.dot_size)

        else:
            max_samples_per_class = self.num_samples // len(classes)

            fig, ax = plt.subplots()

            for cls_idx, cls_id in enumerate(classes):
                this_embed = embed_2d[label == cls_id]
                if this_embed.shape[0] >= max_samples_per_class:
                    perm = torch.randperm(this_embed.shape[0])
                    this_embed = this_embed[perm[:max_samples_per_class]]
                this_embed = torch.transpose(this_embed, 0, 1).cpu()
                ax.scatter(this_embed[0], this_embed[1], self.dot_size)

        fig.savefig(os.path.abspath(self.plot_dir) + f"/{name}_{self.configer.get('iters')}.png")

    def umap(self, embed, label, name):
        label = label.unsqueeze(1).float().clone()
        label = torch.nn.functional.interpolate(label,
                                                 (embed.shape[2], embed.shape[3]), mode='nearest')
        label = label.squeeze(1).long()
        assert label.shape[-1] == embed.shape[-1], '{} {}'.format(label.shape, embed.shape)

        proj_dim = embed.shape[1]
        embed = embed.permute(0, 2, 3, 1)
        embed = embed.contiguous().view(-1, proj_dim)
        label = label.contiguous().view(-1)

        embed_2d = UMAP().fit_transform(embed.cpu())
        embed_2d = torch.from_numpy(embed_2d)

        classes = torch.unique(label)
        classes = [x for x in classes if x != self.ignore_label]

        if self.num_samples == "all":
            fig, ax = plt.subplots()

            for cls_idx, cls_id in enumerate(classes):
                this_embed = embed_2d[label == cls_id]
                this_embed = torch.transpose(this_embed, 0, 1).cpu()
                ax.scatter(this_embed[0], this_embed[1], self.dot_size, label=str(cls_id))

        else:
            max_samples_per_class = self.num_samples // len(classes)

            fig, ax = plt.subplots()

            for cls_idx, cls_id in enumerate(classes):
                this_embed = embed_2d[label == cls_id]
                if this_embed.shape[0] >= max_samples_per_class:
                    perm = torch.randperm(this_embed.shape[0])
                    this_embed = this_embed[perm[:max_samples_per_class]]
                this_embed = torch.transpose(this_embed, 0, 1).cpu()

                if self.configer.get("embedding_visualizer", "save_npy"):
                    np.save(os.path.abspath(self.plot_dir) + f"/{name}_{self.configer.get('iters')}_{cls_id}.npy", this_embed)
                ax.scatter(this_embed[0], this_embed[1], self.dot_size, label=str(cls_id))

        fig.savefig(os.path.abspath(self.plot_dir) + f"/{name}_{self.configer.get('iters')}.png")

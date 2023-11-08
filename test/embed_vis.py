import torch
from lib.vis.embedding_visualizer import EmbeddingVisualizer
import random


class Configer:
    def get(self, *args):
        if args[0] == "embedding_visualizer":
            if args[1] == "num_samples":
                return 100
            elif args[1] == "plot_dir":
                return "/home/jona/PycharmProjects/ContrastiveSeg/test/embeddings/vis_test"
            elif args[1] == "method":
                return "umap"
            elif args[1] == "num_samples":
                return "all"
        elif args[0] == "details":
            return [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
                     [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
                     [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                     [0, 80, 100], [0, 0, 230], [119, 11, 32]]
        elif args[0] == "iters":
            return 0

    def exists(self, *args):
        if args[0] == "embedding_visualizer":
            return True
        return False


configer = Configer()
vis = EmbeddingVisualizer(configer)

center_1 = torch.zeros(20)
center_2 = torch.ones(20)
center_3 = torch.tensor([random.randint(-1, 1) for i in range(20)])
center_4 = torch.tensor([random.randint(-1, 1) for i in range(20)])
centers = [center_1, center_2, center_3, center_4]

embed = []
for center in centers:
    for i in range(10):
        embed.append(center + 0.1 * torch.randn(20))
embed = torch.stack(embed)
embed = embed.view(1, 4, 10, 20)
embed = embed.permute(0, 3, 1, 2)

labels = [i * torch.ones(10) for i in range(4)]
labels = torch.cat(labels)
labels = labels.view(1, 4, 10)

vis.visualize_embeddings(embed, labels)

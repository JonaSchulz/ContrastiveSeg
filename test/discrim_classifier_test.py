import torch
import matplotlib.pyplot as plt

from lib.models.modules.discrim_classifier import DiscrimClassifier


cluster_centers = torch.tensor([[1, 0], [0, 1], [-1, 0]], dtype=torch.float)
classifier = DiscrimClassifier(None, cluster_centers, [1, 2, 3], 1)

embed = []
for c in cluster_centers:
    embed.append(c + 0.1 * torch.randn(8, 2))

embed = torch.cat(embed)

perm = torch.randperm(24)
embed = embed[perm]
embed = embed.view(2, 3, 4, 2)
embed = embed.permute(0, 3, 1, 2)
embed = embed.to(torch.float)

labelmap = torch.argmax(classifier(embed), -1)
labelmap = labelmap.view(-1)
embed = embed.permute(0, 2, 3, 1)
embed = embed.view(-1, 2)

for i in [1, 2, 3]:
    mask = (labelmap == i)
    points = embed[mask]
    plt.scatter(points[:, 0], points[:, 1])

plt.show()

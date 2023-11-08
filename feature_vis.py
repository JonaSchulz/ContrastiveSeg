import torch
import numpy as np
from argparse import ArgumentParser
import os

from lib.utils.tools.configer import Configer
from lib.vis.embedding_visualizer import EmbeddingVisualizer
from lib.utils.tools.logger import Logger as Log


parser = ArgumentParser()
parser.add_argument("--name", type=str, dest="name")
args_parser = parser.parse_args()

configer = Configer(configs="./configs/feature_vis/feature_vis.json")
name = args_parser.name
image = configer.get("source_img")
root_plot_dir = configer.get("embedding_visualizer", "plot_dir")
configer.update(("embedding_visualizer", "plot_dir"), os.path.join(root_plot_dir, name, "embeddings"))
configer.update(("source_dir",), configer.get("source_dir") + name + "/")

Log.info(f"Feature Visualization for {name}")
visualizer = EmbeddingVisualizer(configer)

encode = np.load(configer.get("source_dir") + image + "_encode.npy")
seg = np.load(configer.get("source_dir") + image + "_seg.npy")
embed = np.load(configer.get("source_dir") + image + "_embed.npy")
label = np.load(configer.get("source_dir") + image + "_label.npy")

embed = torch.from_numpy(embed).unsqueeze(0)
encode = torch.from_numpy(encode).unsqueeze(0)
seg = torch.from_numpy(seg).unsqueeze(0)
label = torch.from_numpy(label).unsqueeze(0)

Log.info("Visualize Embeddings...")
visualizer.visualize_embeddings(embed, label, image + "_embed")

# Log.info("Visualize Encodings...")
# visualizer.visualize_embeddings(encode, label, image + "_encode")

Log.info("Visualize Seg Scores...")
visualizer.visualize_embeddings(seg, label, image + "_seg")



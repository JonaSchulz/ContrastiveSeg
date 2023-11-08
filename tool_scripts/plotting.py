import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--smooth", type=int, default=50, dest="smooth")
parser.add_argument("--labels", type=str, dest="labels", nargs="+")
parser.add_argument("--title", type=str, default=None, dest="title")
parser.add_argument("--xlabel", type=str, default="Iteration", dest="xlabel")
parser.add_argument("--ylabel", type=str, default="Loss", dest="ylabel")
parser.add_argument("--save", type=str, default=None, dest="save")
parser.add_argument("--norm", type=bool, default=False, dest="norm")
args_parser = parser.parse_args()

smooth = args_parser.smooth
labels = args_parser.labels
title = args_parser.title
xlabel = args_parser.xlabel
ylabel = args_parser.ylabel
save_path = args_parser.save
norm = args_parser.norm
source_dir = "./plot_source/mem_contrastive"

dataframes = []
for file in sorted(os.listdir(source_dir)):
    dataframes.append(pd.read_csv(os.path.join(source_dir, file)))

vals = []
steps = []
for df in dataframes:
    val = df["value"]
    val = val.rolling(smooth).mean()
    if norm:
        val = (val - val.iloc[-1]) / (val.max() - val.iloc[-1])
    vals.append(val)
    steps.append(df["step"])

plt.style.use("ggplot")
fig, ax = plt.subplots()
for i, val in enumerate(vals):
    if labels is not None:
        ax.plot(steps[i], val, label=labels[i])
    else:
        ax.plot(steps[i], val)
ax.legend()
#ax.set_ylim(bottom=-0.1)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
if title is not None:
    ax.set_title(title)
if save_path is not None:
    fig.savefig(save_path)
plt.show()

import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser


# Extraction function
def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
        return runlog_data
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()


def create_csv(df, path):
    df.to_csv(path + ".csv")


def create_fig(df, name):
    plt.style.use("ggplot")
    fig, ax = plt.subplots()
    ax.set_ylabel(name)
    ax.set_xlabel("Iteration")
    if name != "Learning Rate":
        ax.plot(df["step"], df["value"].rolling(50).mean())
    else:
        ax.plot(df["step"], df["value"])
    return fig


parser = ArgumentParser()
parser.add_argument("--runs", type=str, nargs="+", dest="runs")
parser.add_argument("--fig", type=bool, default=False, dest="fig")
args_parser = parser.parse_args()

runs = args_parser.runs
save_fig = args_parser.fig

root = "../CompleteRunData/"

for run in runs:
    run_root = root + run + "/"
    data = {}
    print(run_root)
    if not os.path.exists(run_root + "plots"):
        os.makedirs(run_root + "plots")
    if os.path.exists(run_root + "tensorboard"):
        print("tensorboard")
        data["tb"] = tflog2pandas(run_root + "tensorboard")
    if os.path.exists(run_root + "loss"):
        print("loss")
        data["log"] = tflog2pandas(run_root + "loss")

    for key, df in data.items():
        if "Loss/Total" in df["metric"].unique() or "Loss_Total" in df["metric"].unique():
            df_ = df.loc[df["metric"] == "Loss/Total"]
            df_ = df_.loc[df_["step"] >= 1000]
            df_ = df_.sort_values("step")
            print(f"plots/loss_total_{key}")
            create_csv(df_, run_root + f"plots/loss_total_{key}")
            if save_fig:
                fig = create_fig(df_, "Seg Loss")
                fig.savefig(run_root + f"plots/loss_total_{key}.png")

        if "Loss/Contrast" in df["metric"].unique():
            df_ = df.loc[df["metric"] == "Loss/Contrast"]
            df_ = df_.loc[df_["step"] >= 1000]
            df_ = df_.sort_values("step")
            print(f"plots/loss_contrast_{key}")
            create_csv(df_, run_root + f"plots/loss_contrast_{key}")
            if save_fig:
                fig = create_fig(df_, "Contrastive Loss")
                fig.savefig(run_root + f"plots/loss_contrast_{key}.png")

        if "Loss/Seg" in df["metric"].unique():
            df_ = df.loc[df["metric"] == "Loss/Seg"]
            df_ = df_.loc[df_["step"] >= 1000]
            df_ = df_.sort_values("step")
            print(f"plots/loss_ce_{key}")
            create_csv(df_, run_root + f"plots/loss_ce_{key}")
            if save_fig:
                fig = create_fig(df_, "CE Loss")
                fig.savefig(run_root + f"plots/loss_ce_{key}.png")

        if "Loss/CE" in df["metric"].unique():
            df_ = df.loc[df["metric"] == "Loss/CE"]
            df_ = df_.loc[df_["step"] >= 1000]
            df_ = df_.sort_values("step")
            print(f"plots/loss_ce_{key}")
            create_csv(df_, run_root + f"plots/loss_ce_{key}")
            if save_fig:
                fig = create_fig(df_, "CE Loss")
                fig.savefig(run_root + f"plots/loss_ce_{key}.png")

        if "Learning Rate" in df["metric"].unique():
            df_ = df.loc[df["metric"] == "Learning Rate"]
            df_ = df_.sort_values("step")
            print("plots/lr_plot")
            create_csv(df_, run_root + f"plots/lr_{key}")
            if save_fig:
                fig = create_fig(df_, "Learning Rate")
                fig.savefig(run_root + "plots/lr_plot.png")

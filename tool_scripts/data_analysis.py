import pandas as pd
import numpy as np

average = []
final = []
for i in range(5):
    ce_loss = pd.read_csv(f"./plot_source/loss_weight_ce/{i}_loss_ce_tb.csv")
    contrastive_loss = pd.read_csv(f"./plot_source/loss_weight_contrastive/{i}_loss_contrast_tb.csv")

    vals_ce = ce_loss["value"]
    vals_contrastive = contrastive_loss["value"]

    vals_ratio = (vals_ce / vals_contrastive).iloc[:].mean()
    vals_ratio_final = (vals_ce / vals_contrastive).iloc[-1]

    average.append(vals_ratio)
    final.append(vals_ratio_final)

    print(f"Average: {vals_ratio}")
    print(f"Final: {vals_ratio_final}")

print(f"Total Average: {sum(average) / len(average)}")
print(f"Total Final: {sum(final) / len(final)}")

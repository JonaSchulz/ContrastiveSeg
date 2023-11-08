import numpy as np

data_ce = np.load("/work/scratch/schulz/ContrastiveSeg/feature_examples/ce/frankfurt_000000_005543_leftImg8bit_seg.npy")
data_contrast = np.load("/work/scratch/schulz/ContrastiveSeg/feature_examples/contrast/frankfurt_000000_005543_leftImg8bit_seg.npy")

label_ce = data_ce.argmax(0)
label_contrast = data_contrast.argmax(0)

np.save("/work/scratch/schulz/ContrastiveSeg/feature_examples/ce/frankfurt_000000_005543_leftImg8bit_predict_label.npy", label_ce)
np.save("/work/scratch/schulz/ContrastiveSeg/feature_examples/contrast/frankfurt_000000_005543_leftImg8bit_predict_label.npy", label_contrast)

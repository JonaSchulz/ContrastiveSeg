import os
import shutil
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--name", type=str, dest="name")
args_parser = parser.parse_args()
name = args_parser.name

log_dir = "/work/scratch/schulz/logs/"
root_dir = "/work/scratch/schulz/ContrastiveSeg/"
run_data_dir = root_dir + "CompleteRunData/"
if not os.path.exists(run_data_dir + name):
    os.makedirs(run_data_dir + name)
    os.makedirs(run_data_dir + name + "/model")
    os.makedirs(run_data_dir + name + "/tensorboard")
run_data_dir = run_data_dir + name + "/"

config = root_dir + "configs/thesis/" + name + ".json"
log = log_dir + name + ".0_err.log"
model_dir = root_dir + "checkpoints/cityscapes/" + name
tensorboard_dir = root_dir + "runs/" + name

shutil.copy(config, run_data_dir)
shutil.copy(log, run_data_dir)
for file in os.listdir(model_dir):
    shutil.copy(model_dir + "/" + file, run_data_dir + "model")
for file in os.listdir(tensorboard_dir):
    shutil.copy(tensorboard_dir + "/" + file, run_data_dir + "tensorboard")

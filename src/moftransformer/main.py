'''
Author: zhangshd
Date: 2024-09-27 23:08:01
LastEditors: zhangshd
LastEditTime: 2024-10-26 16:17:57
'''
from run import main
from run import ex
from config import *
import copy
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_dataset", type=str)
# parser.add_argument("--downstream", type=str, default=None)
parser.add_argument("--log_dir", type=str, default="logs/")
parser.add_argument("--test_only", action="store_true")
parser.add_argument("--seed", type=int, default=42)
# parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--per_gpu_batchsize", type=int)
# parser.add_argument("--num_nodes", type=int, default=1)
# parser.add_argument("--accelerator", type=str, default="auto")
parser.add_argument("--devices", type=int, default=1)
parser.add_argument("--device_vis", type=str)
parser.add_argument("--max_epochs", type=int)
# parser.add_argument("--mean", type=float, default=0.0)
# parser.add_argument("--std", type=float, default=1.0)
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--lr_mult", type=float)
parser.add_argument("--progress_bar", action="store_true")
parser.add_argument("--load_path", type=str)
# parser.add_argument("--n_classes", type=int, default=2)
parser.add_argument("--resume_from", type=str)
parser.add_argument("--task_cfg", type=str, default="test")
parser.add_argument("--model_name", type=str)
parser.add_argument("--limit_train_batches", type=float, nargs="?")
parser.add_argument("--noise_var", type=float, nargs="?")

args = parser.parse_args()

conf = config()
task_conf = eval(args.task_cfg + "()")
conf.update(task_conf)
conf.update({k: v for k, v in vars(args).items() if v not in (None, [], "", "None")})

if "device_vis" in conf and conf["device_vis"] is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = conf["device_vis"]

# ex.run(config_updates=other_args)
main(conf)
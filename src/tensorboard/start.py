#!/usr/bin/python3

# REQUIRES: tensorboard==1.15.0

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
log_dir = dir_path + "/runs"

os.makedirs(log_dir, exist_ok=True)
os.system('tensorboard --logdir="{}"'.format(log_dir))

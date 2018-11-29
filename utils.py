import torch
import torch.nn as nn
import time
import os


def time2str():
    time_id = str(int(time.time()))
    return time_id


def build_dirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        print(e)

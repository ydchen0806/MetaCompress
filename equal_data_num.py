from omegaconf import OmegaConf
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
import codecs
import pandas as pd
import numpy as np
import tifffile


if __name__ == "__main__":

    file_path = "/ssd/0/qjy/Dataset/COVID19_16x128x128_classify_equal/w0_30"
    dataset = ['train', 'val', 'test']
    number = [1000, 100, 10]
    # file_path = "/ssd/0/qjy/Dataset/COVID19_16x128x128_classify_equal/w0_30"
    # dataset = ['train', 'val']
    # number = [1000, 100]

    for num, phase in enumerate(dataset):
        data_root = opj(file_path, phase)
        for idx, data_dir in enumerate(os.listdir(data_root)):
            data_dir = opj(data_root, data_dir)
            if idx >= number[num]:
                os.remove(data_dir)


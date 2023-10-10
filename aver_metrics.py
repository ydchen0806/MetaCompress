from omegaconf import OmegaConf
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
import csv
import codecs
import pandas as pd
import numpy as np
import tifffile


if __name__ == "__main__":

    file_path = "/ssd/0/qjy/MAML/Output/w0_effect2MAML/MAML_30"
    overfit_path = opj(file_path, "overfit")

    steps_num = 21
    train_from_meta = np.zeros(steps_num)
    train_from_meta_classify = np.zeros(steps_num)
    train_from_random = np.zeros(steps_num)

    data_num = 0
    for data_path in os.listdir(overfit_path):
        data_num += 1
        data_path = opj(overfit_path, data_path)
        for method in os.listdir(data_path):
            csv_path = opj(data_path, method, "results.csv")
            with codecs.open(csv_path, encoding='utf-8-sig') as f:
                for idx, row in enumerate(csv.DictReader(f, skipinitialspace=True)):
                    psnr = float(row['psnr'])
                    if method == 'train_from_meta':
                        train_from_meta[idx] += psnr
                    elif method == 'train_from_meta_classify':
                        train_from_meta_classify[idx] += psnr
                    elif method == 'train_from_random':
                        train_from_random[idx] += psnr
    
    train_from_meta = train_from_meta / data_num
    train_from_meta_classify = train_from_meta_classify / data_num
    train_from_random = train_from_random / data_num

    print(f"train_from_meta_classify = {train_from_meta_classify}")
    print(f"train_from_meta = {train_from_meta}")
    print(f"train_from_random = {train_from_random}")








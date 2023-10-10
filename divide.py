import tifffile
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
import random
import shutil
import argparse
from omegaconf import OmegaConf
from utils.misc import configure_optimizer, reconstruct_flattened
from utils.Typing import CompressFrameworkOpt, NormalizeOpt,CropOpt, ReproducOpt, SingleTaskOpt, TransformOpt
from typing import Callable, List, Tuple,Dict, Union
import torch
import torch.optim
from utils.io import *
from utils.transform import *
from utils.dataset import create_flattened_coords
from collections import OrderedDict
import copy
import math
from torch.utils.tensorboard import SummaryWriter
from MAML import FlattenedSampler, FlattenedDataset, Siren, MAML, l2_loss, reproduc, save_model_
import warnings
from torch.optim.lr_scheduler import EPOCH_DEPRECATION_WARNING, ReduceLROnPlateau
from skimage.metrics import structural_similarity
import csv


def crop(file_path, save_path):
    for data_name in os.listdir(file_path):
        data_path = opj(file_path, data_name)
        data = tifffile.imread(data_path)
        d, h, w, _ = data.shape
        mid = d // 2
        if mid >= 16:
            data_crop = data[mid - 16: mid + 16, :, :]
            tifffile.imwrite(opj(save_path, data_name), data_crop)


def random_gen_block(data_path):
    NUMBER_TRAIN = 16000
    NUMBER_VAL = 100
    NUMBER_TEST = 0
    SEED = 42
    DEPTH = 8
    HEIGHT = 128
    WIDTH = 128
    SAVE_PATH_TRAIN = f'/ssd/0/qjy/Dataset/COVID19_CROP_{DEPTH}x{HEIGHT}x{WIDTH}/train'
    SAVE_PATH_VAL = f'/ssd/0/qjy/Dataset/COVID19_CROP_{DEPTH}x{HEIGHT}x{WIDTH}/val'
    SAVE_PATH_TEST = f'/ssd/0/qjy/Dataset/COVID19_CROP_{DEPTH}x{HEIGHT}x{WIDTH}/test'

    random.seed(SEED)

    if os.path.exists(SAVE_PATH_TRAIN):
        shutil.rmtree(SAVE_PATH_TRAIN)
    os.makedirs(SAVE_PATH_TRAIN)

    if os.path.exists(SAVE_PATH_VAL):
        shutil.rmtree(SAVE_PATH_VAL)
    os.makedirs(SAVE_PATH_VAL)

    if os.path.exists(SAVE_PATH_TEST) and NUMBER_TEST != 0:
        shutil.rmtree(SAVE_PATH_TEST)
    if not os.path.exists(SAVE_PATH_TEST):
        os.makedirs(SAVE_PATH_TEST)

    data_list = os.listdir(data_path)
    for i in range(NUMBER_TRAIN + NUMBER_VAL):
        random_data_path = data_list[random.randint(0, len(data_list) - 1)]
        random_data_path = opj(data_path, random_data_path)
        random_data = tifffile.imread(random_data_path)

        data_depth = random_data.shape[0]
        data_height = random_data.shape[1]
        data_width = random_data.shape[2]

        depth_start = random.randint(0, data_depth - DEPTH)
        height_start = random.randint(0, data_height - HEIGHT)
        width_start = random.randint(0, data_width - WIDTH)

        data_cropped = random_data[depth_start:depth_start+DEPTH, height_start:height_start+HEIGHT, width_start:width_start+WIDTH]
        data_name = opb(random_data_path).replace(".tif", "")

        shape = f"_{depth_start}_{height_start}_{width_start}"    # start from 0
        if i < NUMBER_TRAIN:
            tifffile.imwrite(
                opj(SAVE_PATH_TRAIN, data_name + shape + ".tif"),
                data_cropped,
            )
        else:
            tifffile.imwrite(
                opj(SAVE_PATH_VAL, data_name + shape + ".tif"),
                data_cropped,
            )
    
    for i in range(NUMBER_TEST):
        random_data_name = data_list[random.randint(0, len(data_list) - 1)]
        random_data_path = opj(data_path, random_data_name)
        random_data = tifffile.imread(random_data_path)
        tifffile.imwrite(opj(SAVE_PATH_TEST, random_data_name), random_data)


if __name__ == "__main__":

    # crop the original data to the same size (32,512,512)
    file_path = "/ssd/0/qjy/Dataset/COVID19_CT-1"
    save_path = "/ssd/0/qjy/Dataset/COVID19_CROP"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        crop(file_path, save_path)


    # sample block data
    block_save_path = "/ssd/0/qjy/Dataset/COVID19_CROP_BLOCK"
    random_gen_block(save_path)

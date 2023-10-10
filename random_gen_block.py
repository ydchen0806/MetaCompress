import tifffile
import os
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
import random
import shutil


"""
    Randomly generate a specified number of small blocks of specified size
"""

if __name__ == "__main__":
    NUMBER_TRAIN = 12000
    NUMBER_VAL = 2000
    NUMBER_TEST = 200
    SEED = 42
    DEPTH = 16
    HEIGHT = 128
    WIDTH = 128
    PATH = '/ssd/0/qjy/Dataset/COVID19_CT-1'
    SAVE_PATH_TRAIN = f'/ssd/0/qjy/Dataset/COVID19_{DEPTH}x{HEIGHT}x{WIDTH}_equal/train'
    SAVE_PATH_VAL = f'/ssd/0/qjy/Dataset/COVID19_{DEPTH}x{HEIGHT}x{WIDTH}_equal/val'
    SAVE_PATH_TEST = f'/ssd/0/qjy/Dataset/COVID19_{DEPTH}x{HEIGHT}x{WIDTH}_equal/test'

    random.seed(SEED)

    if os.path.exists(SAVE_PATH_TRAIN):
        shutil.rmtree(SAVE_PATH_TRAIN)
    os.makedirs(SAVE_PATH_TRAIN)

    if os.path.exists(SAVE_PATH_VAL):
        shutil.rmtree(SAVE_PATH_VAL)
    os.makedirs(SAVE_PATH_VAL)
        
    if os.path.exists(SAVE_PATH_TEST):
        shutil.rmtree(SAVE_PATH_TEST)
    os.makedirs(SAVE_PATH_TEST)

    data_list = os.listdir(PATH)
    for i in range(NUMBER_TRAIN + NUMBER_VAL + NUMBER_TEST):
        random_data_path = data_list[random.randint(0, len(data_list) - 1)]
        random_data_path = opj(PATH, random_data_path)
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
        elif i >= NUMBER_TRAIN + NUMBER_VAL:
            tifffile.imwrite(
                opj(SAVE_PATH_TEST, data_name + shape + ".tif"),
                data_cropped,
            )
        else:
            tifffile.imwrite(
                opj(SAVE_PATH_VAL, data_name + shape + ".tif"),
                data_cropped,
            )
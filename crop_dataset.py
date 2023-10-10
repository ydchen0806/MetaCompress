import nibabel as nib
import numpy as np
import os
from glob import glob
import argparse
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import re


def read_image(image_path):
    if image_path.endswith('.nii') or image_path.endswith('.nii.gz'):
        image = nib.load(image_path)
        image = image.get_fdata()
    elif image_path.endswith('.png') or image_path.endswith('.jpg') or image_path.endswith('.tif') or image_path.endswith('.tiff'):
        image = cv2.imread(image_path)
    else:
        raise ValueError('Image format not supported.')
    return image

def crop_image(image_path, crop_size, save_path=None):
    img_shape = read_image(image_path).shape
    assert len(img_shape) == len(crop_size), 'Image and crop size must have the same dimension.'
    assert all([img_shape[i] >= crop_size[i] for i in range(len(img_shape))]), 'Crop size must be smaller than image size.'
    # center crop
    crop_start = [int((img_shape[i] - crop_size[i]) / 2) for i in range(len(img_shape))]
    crop_end = [crop_start[i] + crop_size[i] for i in range(len(img_shape))]
    crop_slices = [slice(crop_start[i], crop_end[i]) for i in range(len(img_shape))]
    # crop_slices = [slice(crop_start[i], crop_end[i]) for i in range(3)]
    image = read_image(image_path)[crop_slices]
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        save_name = os.path.basename(image_path).split('.')[0] + '_crop.nii.gz'
        save_path = os.path.join(save_path, save_name)
        if save_path.endswith('.nii') or save_path.endswith('.nii.gz'):
            image = nib.Nifti1Image(image, np.eye(4))
            nib.save(image, save_path)
        elif save_path.endswith('.png') or save_path.endswith('.jpg') or save_path.endswith('.tif') or save_path.endswith('.tiff'):
            cv2.imwrite(save_path, image)
        else:
            raise ValueError('Image format not supported.')
    return image    
    


def match_mask(image_path_list, mask_path_list):
    pattern = r'study_\s*(\d{4})'
    
    # 使用正则表达式模式提取四位数字并创建一个字典，将其映射到路径列表中的元素
    image_dict = {re.search(pattern, x).group(1): x for x in image_path_list}
    mask_dict = {re.search(pattern, x).group(1): x for x in mask_path_list}
    
    # 找到 image 和 mask 共享相同四位数字的元素
    common_four_digits = set(image_dict.keys()) & set(mask_dict.keys())
    
    # 创建结果列表，包含共享相同四位数字的 image 和 mask
    common_image_paths = [image_dict[digit] for digit in common_four_digits]
    common_mask_paths = [mask_dict[digit] for digit in common_four_digits]

    return common_image_paths, common_mask_paths


if __name__ == '__main__':
    import multiprocessing as mp
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default= '/braindat/lab/chenyd/DATASET/COVID19_1110/studies')
    parser.add_argument('--mask_dir', type=str, default = '/braindat/lab/chenyd/DATASET/COVID19_1110/masks')
    parser.add_argument('--crop_size', default=(128,128,16), type=int, nargs='+', help='Crop size.')
    parser.add_argument('--output_dir', type=str, default='/braindat/lab/chenyd/DATASET/COVID19_1110_cropped')
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count() // 2, help='Number of workers.')
    # parser.add_argument('--num_workers', type=int, default=1, help='Number of workers.')
    args = parser.parse_args()
    used_name = 'CT-4'
    image_paths = glob(os.path.join(args.image_dir, used_name,'*nii*'))
    mask_paths = glob(os.path.join(args.mask_dir, '*nii*'))
    output_dir = os.path.join(args.output_dir, used_name)
    os.makedirs(output_dir, exist_ok=True)
    # image_paths, mask_paths = match_mask(image_paths, mask_paths)
    # print('Number of images: {}'.format(len(image_paths)))
    start_time = time.time()
    if args.num_workers == 1:
        for image_path in tqdm(image_paths):
            crop_image(image_path, args.crop_size, output_dir)
    else:
        pool = mp.Pool(args.num_workers)
        for image_path in tqdm(image_paths):
            pool.apply_async(crop_image, args=(image_path, args.crop_size, output_dir))
        pool.close()
        pool.join()  # 等待所有子进程完成
    end_time = time.time()
    print(f'Time elapsed: {end_time - start_time:.2f} s')
    print('Done.')
    num_raw_images = len(image_paths)
    num_cropped_images = len(glob(os.path.join(output_dir, '*nii*')))
    print(f'len of raw images: {num_raw_images}, len of cropped images: {num_cropped_images}')
        
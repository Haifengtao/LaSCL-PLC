#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   h5_test.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/12 14:04   Bot Zhao      1.0         None
"""

# import lib
import sys
import h5py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm
import json

from utils import file_io
from utils import img_utils


def save_h5(times=0):
    if times == 0:
        h5f = h5py.File('data.h5', 'w')
        dataset = h5f.create_dataset("data", (100, 1000, 1000),
                                     maxshape=(None, 1000, 1000),
                                     # chunks=(1, 1000, 1000),
                                     dtype='float32')
    else:
        h5f = h5py.File('data.h5', 'a')
        dataset = h5f['data']
    # 关键：这里的h5f与dataset并不包含真正的数据，
    # 只是包含了数据的相关信息，不会占据内存空间
    #
    # 仅当使用数组索引操作（eg. dataset[0:10]）
    # 或类方法.value（eg. dataset.value() or dataset.[()]）时数据被读入内存中
    a = np.random.rand(100, 1000, 1000).astype('float32')
    # 调整数据预留存储空间（可以一次性调大些）
    dataset.resize([times * 100 + 100, 1000, 1000])
    # 数据被读入内存
    dataset[times * 100:times * 100 + 100] = a
    # print(sys.getsizeof(h5f))
    h5f.close()


def load_h5():
    h5f = h5py.File('data.h5', 'r')
    data = h5f['data'][0:10]
    # print(data)


def save_img_h5(save_name, img, mask, used_z, w, h):
    if used_z == 0:
        h5f = h5py.File(save_name, 'w')
        dataset_img = h5f.create_dataset("img", img.shape, maxshape=(None, w, h), dtype='float16')
    else:
        h5f = h5py.File(save_name, 'a')
        dataset_img = h5f['img']
    z = img.shape[0]
    dataset_img.resize([used_z+z, w, h])          # 调整数据预留存储空间（可以一次性调大些）
    dataset_img[used_z: (used_z+z)] = img         # 数据被读入内存
    if mask is not None:
        if used_z == 0:
            dataset_mask = h5f.create_dataset("mask", mask.shape, maxshape=(None, w, h), dtype='uint8')
        else:
            dataset_mask = h5f['mask']
        dataset_mask.resize([used_z + z, w, h])
        dataset_mask[used_z:(used_z + z)] = mask
    used_z += z
    h5f.close()
    return used_z


def generate_H5_dataset(data_dir, save_name_labeled, save_name_unlabeled):
    """
    generate_H5_dataset (save the raw nii files to H5 file dataset)
    A 2D dataset
    :param data_dir: the statistics data saved at csv file.
    :param save_name_labeled:
    :param save_name_unlabeled:
    :return: file "*/*.h5" saved at save_name_labeled
    /
    /meta_info: a json file; containing {"img_path": path, "mask_path": i[1]["labeled_T1CE"],
                                      "raw_spacing": mask.GetSpacing(), "raw_size": mask.GetSize()}
                and the key value is its uid.
    /img: data [z1+z2+...+zn, 512, 512]
    /mask: data [z1+z2+...+zn, 512, 512]
    the data was padded or resized to (512, 512, None), == (W, H, Z)
    and normalized to 0-1. the max value was set as 3000, and min value was set as 0.
    """
    data = pd.read_csv(data_dir)
    z_nums = []
    meta_info_labeled = {}
    meta_info_unlabeled = {}
    labeled_used_z = 0
    unlabeled_used_z = 0
    for i in tqdm(data.iterrows()):
        if i[1]["T1CE_num"] == 1 and i[1]["labeled_T1CE"] != "0":
            path = i[1]["T1CE"][2:-2]
            img, img_array = file_io.read_array_nii(path)
            mask, mask_array = file_io.read_array_nii(i[1]["labeled_T1CE"])
            if len(img_array.shape) > 3:
                continue
            # no crop: Because the background is not the value zero.
            z, y, x = img_array.shape
            # pad and resize
            if x <= 512 and y <= 512:
                resize_img = np.pad(img_array, (
                (0, 0), ((512 - y) // 2, (512 - y) - (512 - y) // 2), ((512 - x) // 2, (512 - x) - (512 - x) // 2),),
                                    "constant", constant_values=((0, 0), (0, 0), (0, 0)))
                resize_mask = np.pad(mask_array, (
                (0, 0), ((512 - y) // 2, (512 - y) - (512 - y) // 2), ((512 - x) // 2, (512 - x) - (512 - x) // 2)),
                                     "constant", constant_values=((0, 0), (0, 0), (0, 0)))
            else:
                temp_img = img_utils.resize_image_size(img, (512, 512, z), resamplemethod=sitk.sitkLinear)
                resize_img = sitk.GetArrayFromImage(temp_img)
                temp_mask = img_utils.resize_image_size(mask, (512, 512, z))
                resize_mask = sitk.GetArrayFromImage(temp_mask)
            norm_arr = img_utils.normalize_0_1(resize_img, min_intensity=0, max_intensity=3000)
            save_img_h5(save_name_labeled, norm_arr, resize_mask, labeled_used_z, 512, 512)
            labeled_used_z += z
            z_nums.append(z)
            meta_info_labeled[i[1]["uid"]] = {"img_path": path, "mask_path": i[1]["labeled_T1CE"],
                                      "raw_spacing": mask.GetSpacing(), "raw_size": mask.GetSize()}
        elif i[1]["T1CE_num"] == 1 and i[1]["labeled_T1CE"] == "0":
            path = i[1]["T1CE"][2:-2]
            img, img_array = file_io.read_array_nii(path)
            if len(img_array.shape) > 3:
                continue
            # no crop
            z, y, x = img_array.shape
            # pad and resize
            if x <= 512 and y <= 512:
                resize_img = np.pad(img_array, (
                    (0, 0), ((512 - y) // 2, (512 - y) - (512 - y) // 2),
                    ((512 - x) // 2, (512 - x) - (512 - x) // 2),),
                                    "constant", constant_values=((0, 0), (0, 0), (0, 0)))
            else:
                temp_img = img_utils.resize_image_size(img, (512, 512, z), resamplemethod=sitk.sitkLinear)
                resize_img = sitk.GetArrayFromImage(temp_img)
            norm_arr = img_utils.normalize_0_1(resize_img, min_intensity=0, max_intensity=3000)
            save_img_h5(save_name_unlabeled, norm_arr, None, unlabeled_used_z, 512, 512)
            unlabeled_used_z += z
            z_nums.append(z)
            meta_info_unlabeled[i[1]["uid"]] = {"img_path": path, "mask_path": None,
                                                "raw_spacing": img.GetSpacing(), "raw_size": img.GetSize()}

    meta_info_labeled = json.dumps(meta_info_labeled)
    if not os.path.exists(save_name_labeled):
        f = h5py.File(save_name_labeled, 'w')
    else:
        f = h5py.File(save_name_labeled, 'a')
    f.create_dataset('meta_info', data=meta_info_labeled)
    f.close()

    meta_info_unlabeled = json.dumps(meta_info_unlabeled)
    if not os.path.exists(save_name_unlabeled):
        f = h5py.File(save_name_unlabeled, 'w')
    else:
        f = h5py.File(save_name_unlabeled, 'a')
    f.create_dataset('meta_info', data=meta_info_unlabeled)
    f.close()


#######################
# TEST
#######################
# if __name__ == '__main__':
#     # Created
#     data_dir = r"D:\pycharm_project\tumor_classification\data\dataset1_summary_v1.csv"
#     save_dir_label = r"D:\dataset\tumor_classification\dataset1_labeled.h5"
#     save_dir_unlabeled = r"D:\dataset\tumor_classification\dataset1_unlabeled.h5"
#     generate_H5_dataset(data_dir, save_dir_label, save_dir_unlabeled)
#
#     # lOAD
#     data_dir = r"D:\dataset\tumor_classification\preprocessed_dataset1\dataset1_unlabeled.h5"
#     f = h5py.File(data_dir, 'r')
#     print(f.keys())
#     data = f['meta_info']
#     print(data.keys)
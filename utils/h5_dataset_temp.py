#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   h5_dataset.py
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
sys.path.append("./")
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


def save_img_h5(save_name, img, mask, label, used_z, w, h):
    if used_z == 0:
        h5f = h5py.File(save_name, 'w')
        dataset_img = h5f.create_dataset("img", img.shape, maxshape=(None, w, h), dtype='float32')
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

    if label is not None:
        if used_z == 0:
            dataset_labels = h5f.create_dataset("label", label.shape, maxshape=(None, ), dtype='uint8')
        else:
            dataset_labels = h5f['label']
        dataset_labels.resize([used_z + z])
        dataset_labels[used_z:(used_z + z)] = label
    used_z += z
    h5f.close()
    return used_z


def generate_H5_dataset(data_dir, save_name_labeled_train, save_name_labeled_test, save_name_labeled_val, save_name_unlabeled, indices):
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
    meta_info_labeled_train = {}
    meta_info_labeled_test = {}
    meta_info_labeled_val = {}

    meta_info_unlabeled = {}

    labeled_used_z_train = 0
    labeled_used_z_test = 0
    labeled_used_z_val = 0

    unlabeled_used_z = 0
    idx = 0
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

            if idx in indices[0]:
                save_img_h5(save_name_labeled_train, norm_arr, resize_mask, labeled_used_z_train, 512, 512)
                labeled_used_z_train += z
                meta_info_labeled_train[i[1]["uid"]] = {"img_path": path, "mask_path": i[1]["labeled_T1CE"],
                                                        "raw_spacing": mask.GetSpacing(),
                                                        "raw_origin": mask.GetOrigin(),
                                                        "raw_direction": mask.GetDirection(),
                                                        "raw_size": mask.GetSize()}
            elif idx in indices[1]:
                save_img_h5(save_name_labeled_val, norm_arr, resize_mask, labeled_used_z_val, 512, 512)
                labeled_used_z_val += z
                meta_info_labeled_val[i[1]["uid"]] = {"img_path": path, "mask_path": i[1]["labeled_T1CE"],
                                                      "raw_spacing": mask.GetSpacing(),
                                                      "raw_origin": mask.GetOrigin(),
                                                      "raw_direction": mask.GetDirection(),
                                                      "raw_size": mask.GetSize()}
            else:
                save_img_h5(save_name_labeled_test, norm_arr, resize_mask, labeled_used_z_test, 512, 512)
                labeled_used_z_test += z
                meta_info_labeled_test[i[1]["uid"]] = {"img_path": path, "mask_path": i[1]["labeled_T1CE"],
                                                       "raw_spacing": mask.GetSpacing(),
                                                       "raw_origin": mask.GetOrigin(),
                                                       "raw_direction": mask.GetDirection(),
                                                       "raw_size": mask.GetSize()}
            idx += 1
        elif i[1]["T1CE_num"] == 1 and i[1]["labeled_T1CE"] == "0":
            continue
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

    meta_info_labeled_train = json.dumps(meta_info_labeled_train)
    if not os.path.exists(save_name_labeled_train):
        f = h5py.File(save_name_labeled_train, 'w')
    else:
        f = h5py.File(save_name_labeled_train, 'a')
    f.create_dataset('meta_info', data=meta_info_labeled_train)
    f.close()

    meta_info_labeled_test = json.dumps(meta_info_labeled_test)
    if not os.path.exists(save_name_labeled_test):
        f = h5py.File(save_name_labeled_test, 'w')
    else:
        f = h5py.File(save_name_labeled_test, 'a')
    f.create_dataset('meta_info', data=meta_info_labeled_test)
    f.close()

    meta_info_unlabeled = json.dumps(meta_info_unlabeled)
    if not os.path.exists(save_name_unlabeled):
        f = h5py.File(save_name_unlabeled, 'w')
    else:
        f = h5py.File(save_name_unlabeled, 'a')
    f.create_dataset('meta_info', data=meta_info_unlabeled)
    f.close()


def generate_H5_dataset_v2(data_dir, save_name_labeled_train, save_name_labeled_test, save_name_labeled_val, save_name_unlabeled, indices):
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
    meta_info_labeled_train = {}
    meta_info_labeled_test = {}
    meta_info_labeled_val = {}

    meta_info_unlabeled = {}

    labeled_used_z_train = 0
    labeled_used_z_test = 0
    labeled_used_z_val = 0

    unlabeled_used_z = 0
    idx = 0

    f_train = open("../data/train_fine.txt", "w")
    f_test = open("../data/train_test.txt", "w")
    f_val = open("../data/train_val.txt", "w")
    for i in tqdm(data.iterrows()):
        if i[1]["T1CE_num"] == 1 and i[1]["labeled_T1CE"] != "0":
            path = i[1]["T1CE"][2:-2]
            img, img_array = file_io.read_array_nii(path)
            mask, mask_array = file_io.read_array_nii(i[1]["labeled_T1CE"])
            if len(img_array.shape) > 3:
                continue

            # 1. remove small cc
            mask_arr = img_utils.rm_small_cc(mask_array.astype("int32"), rate=0.5)

            # 2. crop by mask
            st, ed = img_utils.get_bbox(mask_arr)
            img_roi = img_utils.crop_img(img_array, st, ed)
            mask_roi = img_utils.crop_img(mask_array, st, ed)
            # no crop: Because the background is not the value zero.
            z, y, x = img_roi.shape
            # pad and resize
            max_x, max_y = 208, 208
            if x <= max_x and y <= max_y:
                resize_img = np.pad(img_roi, (
                (0, 0), ((max_y - y) // 2, (max_y - y) - (max_y - y) // 2), ((max_x - x) // 2, (max_x - x) - (max_x - x) // 2),),
                                    "constant", constant_values=((0, 0), (0, 0), (0, 0)))
                resize_mask = np.pad(mask_roi, (
                (0, 0), ((max_y - y) // 2, (max_y - y) - (max_y - y) // 2), ((max_x - x) // 2, (max_x - x) - (max_x - x) // 2)),
                                     "constant", constant_values=((0, 0), (0, 0), (0, 0)))
            else:
                resize_img = img_utils.resize_3d_arr(img_roi, (z, max_y, max_x), order=2)
                resize_mask = img_utils.resize_3d_arr(mask_roi, (z, max_y, max_x), order=0)
            # norm_arr = img_utils.normalize_0_1(resize_img, min_intensity=0, max_intensity=3000)
            if idx in indices[0]:
                save_img_h5(save_name_labeled_train, norm_arr, resize_mask, labeled_used_z_train, max_x, max_y)
                labeled_used_z_train += z
                meta_info_labeled_train[i[1]["uid"]] = {"img_path": path, "mask_path": i[1]["labeled_T1CE"],
                                                        "raw_spacing": mask.GetSpacing(),
                                                        "raw_origin": mask.GetOrigin(),
                                                        "raw_direction": mask.GetDirection(),
                                                        "raw_size": mask.GetSize()}
                f_train.write(path + '\n')
            elif idx in indices[1]:
                save_img_h5(save_name_labeled_val, norm_arr, resize_mask, labeled_used_z_val, max_x, max_y)
                labeled_used_z_val += z
                meta_info_labeled_val[i[1]["uid"]] = {"img_path": path, "mask_path": i[1]["labeled_T1CE"],
                                                      "raw_spacing": mask.GetSpacing(),
                                                      "raw_origin": mask.GetOrigin(),
                                                      "raw_direction": mask.GetDirection(),
                                                      "raw_size": mask.GetSize()}
                f_val.write(path + '\n')
            else:
                save_img_h5(save_name_labeled_test, norm_arr, resize_mask, labeled_used_z_test, max_x, max_y)
                labeled_used_z_test += z
                meta_info_labeled_test[i[1]["uid"]] = {"img_path": path, "mask_path": i[1]["labeled_T1CE"],
                                                       "raw_spacing": mask.GetSpacing(),
                                                       "raw_origin": mask.GetOrigin(),
                                                       "raw_direction": mask.GetDirection(),
                                                       "raw_size": mask.GetSize()}
                f_test.write(path + '\n')
            idx += 1
        elif i[1]["T1CE_num"] == 1 and i[1]["labeled_T1CE"] == "0":
            continue
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
            # norm_arr = img_utils.normalize_0_1(resize_img, min_intensity=0, max_intensity=3000)
            save_img_h5(save_name_unlabeled, norm_arr, None, unlabeled_used_z, 512, 512)
            unlabeled_used_z += z
            z_nums.append(z)
            meta_info_unlabeled[i[1]["uid"]] = {"img_path": path, "mask_path": None,
                                                "raw_spacing": img.GetSpacing(), "raw_size": img.GetSize()}

    meta_info_labeled_train = json.dumps(meta_info_labeled_train)
    if not os.path.exists(save_name_labeled_train):
        f = h5py.File(save_name_labeled_train, 'w')
    else:
        f = h5py.File(save_name_labeled_train, 'a')
    f.create_dataset('meta_info', data=meta_info_labeled_train)
    f.close()

    meta_info_labeled_test = json.dumps(meta_info_labeled_test)
    if not os.path.exists(save_name_labeled_test):
        f = h5py.File(save_name_labeled_test, 'w')
    else:
        f = h5py.File(save_name_labeled_test, 'a')
    f.create_dataset('meta_info', data=meta_info_labeled_test)
    f.close()

    # meta_info_unlabeled = json.dumps(meta_info_unlabeled)
    # if not os.path.exists(save_name_unlabeled):
    #     f = h5py.File(save_name_unlabeled, 'w')
    # else:
    #     f = h5py.File(save_name_unlabeled, 'a')
    # f.create_dataset('meta_info', data=meta_info_unlabeled)
    # f.close()


def generate_H5_dataset_seg2cls(data_dir, save_name, pad=3):
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
    with open(data_dir, 'r') as f:
        # import pdb
        # pdb.set_trace()
        strs = f.readlines()
        img_list = [line.strip().split(" ")[0] for line in strs]
        mask_list = [line.strip().split(" ")[1] for line in strs]
    meta_info_labeled_train = []
    labeled_used_z_train = 0
    idx = 0
    for img_path, mask_path in zip(img_list, mask_list):
        # try:
        img, img_arr = file_io.read_array_nii(img_path.replace("_roi_resize_crop.nii.gz", ".nii.gz").replace("zhang_istbi", "zhang_istbi/data_disk"))
        mask, mask_arr = file_io.read_array_nii(mask_path.replace("_roi_resize_crop.nii.gz", ".nii.gz").replace("zhang_istbi", "zhang_istbi/data_disk"))
        mask_arr[mask_arr!=1] = 0
        # 1. crop image
        mask_indexs = np.nonzero(mask_arr)
        minz, maxz = np.min(mask_indexs[0]), np.max(mask_indexs[0])
        st_z, ed_z = np.max([0, minz-pad]), np.min([maxz+pad, mask_arr.shape[0]])

        # 2. crop by mask
        st, ed = [st_z, 0, 0], [ed_z, mask_arr.shape[1], mask_arr.shape[2]]
        img_roi = img_utils.crop_img(img_arr, st, ed)
        mask_roi = img_utils.crop_img(mask_arr, st, ed)

        # 3. resample to the same spacing
        img_roi_img = sitk.GetImageFromArray(img_roi)
        mask_roi_img = sitk.GetImageFromArray(mask_roi)
        mask_roi_img.SetSpacing(mask.GetSpacing())
        img_roi_img.SetSpacing(img.GetSpacing())
        img_itkimg_Resampled = img_utils.resize_image_itk(img_roi_img, (1, 1, mask.GetSpacing()[2]), resamplemethod=sitk.sitkLinear)
        mask_itkimg_Resampled = img_utils.resize_image_itk(mask_roi_img, (1, 1, mask.GetSpacing()[2]), resamplemethod=sitk.sitkLinear)
        img_Resampled = sitk.GetArrayFromImage(img_itkimg_Resampled)
        mask_Resampled = sitk.GetArrayFromImage(mask_itkimg_Resampled)

        # 4. crop or padding to the same size
        z, y, x = img_Resampled.shape
        # pad and crop
        max_x, max_y, max_z = 256, 256, 64

        if y <= max_y:
            cropped_img = np.pad(img_Resampled, ((0, 0), ((max_y - y) // 2, (max_y - y) - (max_y - y) // 2),
                                                 (0, 0),), "constant", constant_values=((0, 0), (0, 0), (0, 0)))
            cropped_mask = np.pad(mask_Resampled, ((0, 0), ((max_y - y) // 2, (max_y - y) - (max_y - y) // 2),
                                                   (0, 0),), "constant", constant_values=((0, 0), (0, 0), (0, 0)))
        else:
            st_y = (y - max_y) // 2
            cropped_img = img_Resampled[:, st_y:(st_y + max_y), :]
            cropped_mask = mask_Resampled[:, st_y:(st_y + max_y), :]

        if x <= max_x:
            cropped_img = np.pad(cropped_img, ((0, 0), (0, 0),
                                               ((max_x - x) // 2, (max_x - x) - (max_x - x) // 2)), "constant",
                                 constant_values=((0, 0), (0, 0), (0, 0)))
            cropped_mask = np.pad(cropped_mask, ((0, 0), (0, 0),
                                                 ((max_x - x) // 2, (max_x - x) - (max_x - x) // 2)), "constant",
                                  constant_values=((0, 0), (0, 0), (0, 0)))
        else:
            st_x = (x - max_x) // 2
            cropped_img = cropped_img[..., st_x:(st_x + max_x)]
            cropped_mask = cropped_mask[..., st_x:(st_x + max_x)]

        # 5. save h5 file
        if "hemangioblastoma" in img_path:  # 血管细胞瘤
            labels = np.array([1] * cropped_img.shape[0])
        elif "angiocavernoma" in img_path:  # 海绵状血管瘤
            labels = np.array([0] * cropped_img.shape[0])
        elif "III-IV_glioma" in img_path:  # 3-4型胶质瘤
            labels = np.array([2] * cropped_img.shape[0])
        elif "pilocytic_astrocytoma" in img_path:
            labels = np.array([3] * cropped_img.shape[0])
        else:
            raise Exception("Error path")
        for i in range(cropped_img.shape[0]):
            if np.max(cropped_img[i, ...]) == 0:
                print(img_path)

        # cropped_img = img_utils.normalize_0_1(cropped_img, min_intensity=0, max_intensity=3000)
        save_img_h5(save_name, cropped_img, cropped_mask, labels, labeled_used_z_train, 256, 256)
        labeled_used_z_train = labeled_used_z_train + cropped_img.shape[0]
        meta_info_labeled_train += [idx] * cropped_img.shape[0]
        idx += 1
    h5f = h5py.File(save_name, 'a')
    h5f.create_dataset('meta_info', data=meta_info_labeled_train)
    h5f.close()


def generate_H5_dataset_seg2cls_biasC(data_dir, save_name, pad=3):
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
    with open(data_dir, 'r') as f:
        strs = f.readlines()
        img_list = [line.strip().split(" ")[0][3:] for line in strs]
        mask_list = [line.strip().split(" ")[1][3:] for line in strs]
    meta_info_labeled_train = []
    labeled_used_z_train = 0
    idx = 0

    save_dir, _ = os.path.split(save_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for img_path, mask_path in zip(img_list, mask_list):
        # try:
        img, img_arr = file_io.read_array_nii(img_path)
        mask, mask_arr = file_io.read_array_nii(mask_path)
        if img_arr.shape != mask_arr.shape:
            print(img_arr.shape, img_path)

        tumor_labels = list(np.unique(mask_arr))
        if len(tumor_labels) == 2:
            tumor_label = 1
        else:
            # print()
            max_num = np.inf
            tumor_label = np.max(tumor_labels)
            for i in tumor_labels:
                if i == 0:
                    continue
                temp = np.zeros(mask_arr.shape)
                temp[mask_arr == i] = 1
                nums = np.sum(temp)
                if nums < max_num:
                    tumor_label = i
            print(tumor_label, labeled_used_z_train, mask_path)


        # 1. resample to the same spacing
        img_itkimg_Resampled = img_utils.resize_image_itk(img, (1, 1, 1), resamplemethod=sitk.sitkLinear)
        mask_itkimg_Resampled = img_utils.resize_image_itk(mask, (1, 1, 1), resamplemethod=sitk.sitkNearestNeighbor)
        img_Resampled = sitk.GetArrayFromImage(img_itkimg_Resampled)
        mask_Resampled = sitk.GetArrayFromImage(mask_itkimg_Resampled)

        # 2. crop image
        mask_Resampled[mask_Resampled != tumor_label] = 0
        mask_Resampled[mask_Resampled == tumor_label] = 1
        mask_indexs = np.nonzero(mask_Resampled)
        minz, maxz = np.min(mask_indexs[0]), np.max(mask_indexs[0])
        # st_z, ed_z = np.max([0, minz - pad]), np.min([maxz + pad, mask_arr.shape[0]])

        # 3. crop by mask
        if maxz - minz + 1 < 32:
            new_pad1 = (32-(maxz-minz+1))//2
            new_pad2 = (32-(maxz-minz+1))-new_pad1
            if minz - new_pad2 < 0:
                new_pad2 = minz
                new_pad1 = (32-(maxz-minz+1))-new_pad2
            st_z, ed_z = np.max([0, minz - new_pad2]), np.min([maxz + new_pad1, mask_Resampled.shape[0]])
        else:
            new_pad1 = ((maxz-minz+1)-32)//2
            new_pad2 = ((maxz-minz+1)-32)-new_pad1
            st_z, ed_z = np.max([0, minz + new_pad2]), np.min([maxz - new_pad1, mask_Resampled.shape[0]])

        st, ed = [st_z, 0, 0], [ed_z, mask_Resampled.shape[1], mask_Resampled.shape[2]]
        img_roi = img_utils.crop_img(img_Resampled, st, ed)
        mask_roi = img_utils.crop_img(mask_Resampled, st, ed)
        if mask_roi.shape[0] != 32:
            import pdb
            pdb.set_trace()
        # 4. crop or padding to the same size
        z, y, x = img_roi.shape
        # pad and crop
        max_x, max_y, max_z = 256, 256, 32

        if y <= max_y:
            cropped_img = np.pad(img_roi, ((0, 0), ((max_y - y) // 2, (max_y - y) - (max_y - y) // 2),
                                                 (0, 0),), "constant", constant_values=((0, 0), (0, 0), (0, 0)))
            cropped_mask = np.pad(mask_roi, ((0, 0), ((max_y - y) // 2, (max_y - y) - (max_y - y) // 2),
                                                   (0, 0),), "constant", constant_values=((0, 0), (0, 0), (0, 0)))
        else:
            st_y = (y - max_y) // 2
            cropped_img = img_roi[:, st_y:(st_y + max_y), :]
            cropped_mask = mask_roi[:, st_y:(st_y + max_y), :]

        if x <= max_x:
            cropped_img = np.pad(cropped_img, ((0, 0), (0, 0),
                                               ((max_x - x) // 2, (max_x - x) - (max_x - x) // 2)), "constant",
                                 constant_values=((0, 0), (0, 0), (0, 0)))
            cropped_mask = np.pad(cropped_mask, ((0, 0), (0, 0),
                                                 ((max_x - x) // 2, (max_x - x) - (max_x - x) // 2)), "constant",
                                  constant_values=((0, 0), (0, 0), (0, 0)))
        else:
            st_x = (x - max_x) // 2
            cropped_img = cropped_img[..., st_x:(st_x + max_x)]
            cropped_mask = cropped_mask[..., st_x:(st_x + max_x)]

        # 5. save h5 file
        if "hemangioblastoma" in img_path:  # 血管细胞瘤
            labels = np.array([1] * cropped_img.shape[0])
        elif "angiocavernoma" in img_path:  # 海绵状血管瘤
            labels = np.array([0] * cropped_img.shape[0])
        elif "III-IV_glioma" in img_path:  # 3-4型胶质瘤
            labels = np.array([2] * cropped_img.shape[0])
        elif "pilocytic_astrocytoma" in img_path:
            labels = np.array([3] * cropped_img.shape[0])
        else:
            raise Exception("Error path")
        for i in range(cropped_img.shape[0]):
            if np.max(cropped_img[i, ...]) == 0:
                print("==>error", img_path)

        # cropped_img = img_utils.normalize_0_1(cropped_img, min_intensity=0, max_intensity=3000)

        save_img_h5(save_name, cropped_img, cropped_mask, labels, labeled_used_z_train, 256, 256)
        labeled_used_z_train = labeled_used_z_train + cropped_img.shape[0]
        meta_info_labeled_train += [idx] * cropped_img.shape[0]
        idx += 1
    h5f = h5py.File(save_name, 'a')
    h5f.create_dataset('meta_info', data=meta_info_labeled_train)
    h5f.close()


if __name__ == '__main__':
    datasets = ['fold1', 'fold2', 'fold3', "fold4", "fold5"]
    for i in datasets:
        print("==> runningg {}".format(i))
        # data_dir = "./data/version_final_5flod/"+i+"/test.txt"
        # save_name = "../../dataset/tumor_cls/cross_val_h5_dataset_baisC_resmple/"+i+"_2d_final_test.h5"
        # generate_H5_dataset_seg2cls_biasC(data_dir, save_name, pad=3)

        data_dir = "./data/version_final_5flod/" + i + "/train.txt"
        save_name = "../../dataset/tumor_cls/cross_val_h5_dataset_baisC_resmple/"+i+"_2d_final_train.h5"
        generate_H5_dataset_seg2cls_biasC(data_dir, save_name, pad=3)
#######################
# TEST
#######################
# if __name__ == '__main__':
#     # Created
#     data_dir = r"D:\pycharm_project\tumor_classification\data\dataset1_summary_v1.csv"
#     save_dir_unlabeled = r"D:\dataset\tumor_classification\dataset1_unlabeled.h5"
#
#     save_dir_train = r"D:\dataset\tumor_classification\preprocessed_dataset1\dataset1_labeled_fine_train_v2.h5"
#     data_test = r"D:\dataset\tumor_classification\preprocessed_dataset1\dataset1_labeled_fine_test_v2.h5"
#     save_dir_val = r"D:\dataset\tumor_classification\preprocessed_dataset1\dataset1_labeled_fine_val_v2.h5"
#     arr = np.arange(30)
#     np.random.shuffle(arr)
#     train_indices = arr[:22]
#     test_indexs = arr[22:27]
#     val_indexs = arr[27:]
#     print(test_indexs)
#     print(val_indexs)
#     print(train_indices)
#     generate_H5_dataset_v2(data_dir, save_dir_train, data_test, save_dir_val, save_dir_unlabeled,
#                            [train_indices, val_indexs, test_indexs])
#     labeled_f_val = h5py.File(save_dir_train, 'r')
#
#     # import pdb
#     # pdb.set_trace()
#     # data_info = labeled_f_val['meta_info']
#     # print(labeled_f_val)
#     # lOAD
#     data_dir = r"D:\dataset\tumor_classification\preprocessed_dataset1\dataset1_labeled.h5"
#
#     f = h5py.File(data_dir, 'r')
#     w = h5py.File(data_test, 'w')
#     print(f.keys())
#     img = f['img']
#     mask = f["mask"]
#     meta_info = f["meta_info"]
#
#
#     w.create_dataset("data_test")
#     print(data.keys)

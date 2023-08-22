#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   data_generator.py
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/5 16:44   Bot Zhao      1.0         None
"""

# import lib
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import torch
import nibabel as nib
# from torchvision import transforms
import h5py, sys
from PIL import Image, ImageOps, ImageFilter

# sys.path.append("/home1/zhaobt/PycharmProjects/tumor_cls/")
sys.path.append("/data/data_disk/zhaobt/project/tumor_cls/")
from utils import file_io, augment
import os, json
from utils import img_utils
import random

# import nibabel
# from scipy import ndimage

from torch.utils.data import DataLoader, Dataset


class cls_generator_2D(Dataset):
    def __init__(self, labeled_img, labeled_mask, labels, rotate_degree=15, noise_sigma=(0.001, 0.002)):
        self.labeled_img = labeled_img
        self.labeled_mask = labeled_mask
        self.labels = labels
        self.degree = rotate_degree
        self.noise_sigma = noise_sigma

    def __len__(self):
        return self.labeled_img.shape[0]

    def __getitem__(self, index):
        X = self.labeled_img[index, ...]
        mask = self.labeled_mask[index, ...]
        X = img_utils.normalize_0_1(X)
        # 0. Rotate
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        X = Image.fromarray(np.uint8(X * 255))
        mask = Image.fromarray(np.uint8(mask))
        X = X.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        X = np.array(X)[np.newaxis, ...].astype(np.float32) / 255
        mask = np.array(mask)
        #
        # 1. Noise
        c, h, w = X.shape
        random_sigma = (np.random.rand(1) * (self.noise_sigma[1] - self.noise_sigma[0])) + self.noise_sigma[0]
        X += np.random.randn(c, h, w) * random_sigma

        # 2. RandomFlip
        if np.random.choice([0, 1]):
            X = X[:, :, ::-1]
            mask = mask[:, ::-1]

        if np.random.choice([0, 1]):
            X = X[:, ::-1, :]
            mask = mask[::-1, :]

        label = np.array(self.labels[index, ...])
        return torch.from_numpy(X.astype(np.float32)), \
               torch.from_numpy(mask.astype(np.uint8)).long(), \
               torch.from_numpy(label.astype(np.uint8)).long()


class cls_val_2D(Dataset):
    def __init__(self, labeled_img, labeled_mask, labels, rotate_degree=15, noise_sigma=(0.001, 0.002)):
        self.labeled_img = labeled_img
        self.labeled_mask = labeled_mask
        self.labels = labels
        self.degree = rotate_degree
        self.noise_sigma = noise_sigma

    def __len__(self):
        return self.labeled_img.shape[0]

    def __getitem__(self, index):
        X = self.labeled_img[index, ...]
        mask = self.labeled_mask[index, ...]
        X = img_utils.normalize_0_1(X)
        # 0. Rotate
        # rotate_degree = random.uniform(-1 * self.degree, self.degree)
        # X = Image.fromarray(np.uint8(X * 255))
        # mask = Image.fromarray(np.uint8(mask))
        # X = X.rotate(rotate_degree, Image.BILINEAR)
        # mask = mask.rotate(rotate_degree, Image.NEAREST)
        X = np.array(X)[np.newaxis, ...].astype(np.float32) # / 255
        mask = np.array(mask)
        #
        # # 1. Noise
        # c, h, w = X.shape
        # random_sigma = (np.random.rand(1) * (self.noise_sigma[1] - self.noise_sigma[0])) + self.noise_sigma[0]
        # X += np.random.randn(c, h, w) * random_sigma
        #
        # # 2. RandomFlip
        # if np.random.choice([0, 1]):
        #     X = X[:, :, ::-1]
        #     mask = mask[:, ::-1]
        #
        # if np.random.choice([0, 1]):
        #     X = X[:, ::-1, :]
        #     mask = mask[::-1, :]

        label = np.array(self.labels[index, ...])
        return torch.from_numpy(X.astype(np.float32)), \
               torch.from_numpy(mask.astype(np.uint8)).long(), \
               torch.from_numpy(label.astype(np.uint8)).long()


class cls_generator_2D_age(Dataset):
    def __init__(self, labeled_img, labeled_mask, labels, ages, sexes, rotate_degree=15,
                 RandomFlip=False,
                 noise_sigma=(0.001, 0.002)):
        self.labeled_img = labeled_img
        self.labeled_mask = labeled_mask
        self.labels = labels
        self.ages = ages
        self.sexes = sexes
        self.degree = rotate_degree
        self.noise_sigma = noise_sigma
        self.RandomFlip = RandomFlip

    def __len__(self):
        return self.labeled_img.shape[0]

    def __getitem__(self, index):
        X = self.labeled_img[index, ...]
        mask = self.labeled_mask[index, ...]
        X = img_utils.normalize_0_1(X)
        # 0. Rotate
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        X = Image.fromarray(np.uint8(X * 255))
        mask = Image.fromarray(np.uint8(mask))
        X = X.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        X = np.array(X)[np.newaxis, ...].astype(np.float32) / 255
        mask = np.array(mask)
        #
        # 1. Noise
        c, h, w = X.shape
        random_sigma = (np.random.rand(1) * (self.noise_sigma[1] - self.noise_sigma[0])) + self.noise_sigma[0]
        X += np.random.randn(c, h, w) * random_sigma

        # 2. RandomFlip
        if self.RandomFlip:
            if np.random.choice([0, 1]):
                X = X[:, :, ::-1]
                mask = mask[:, ::-1]

            if np.random.choice([0, 1]):
                X = X[:, ::-1, :]
                mask = mask[::-1, :]

        label = np.array(self.labels[index, ...])
        # import pdb
        # pdb.set_trace()
        age = np.array(self.ages[index, ...] / 75)
        # print(age)
        sex = np.array(self.sexes[index, ...])


        return torch.from_numpy(X.astype(np.float32)), \
               torch.from_numpy(mask.astype(np.uint8)).long(), \
               torch.from_numpy(label.astype(np.uint8)).long(),\
               torch.from_numpy(age.astype(np.float32)), \
               torch.from_numpy(sex.astype(np.uint8)).long()


class cls_val_2D_age(Dataset):
    def __init__(self, labeled_img, labeled_mask, labels, ages, sexes):
        self.labeled_img = labeled_img
        self.labeled_mask = labeled_mask
        self.labels = labels
        self.ages = ages
        self.sexes = sexes

    def __len__(self):
        return self.labeled_img.shape[0]

    def __getitem__(self, index):
        X = self.labeled_img[index, ...]
        mask = self.labeled_mask[index, ...]
        X = img_utils.normalize_0_1(X)
        # 0. Rotate
        # rotate_degree = random.uniform(-1 * self.degree, self.degree)
        # X = Image.fromarray(np.uint8(X * 255))
        # mask = Image.fromarray(np.uint8(mask))
        # X = X.rotate(rotate_degree, Image.BILINEAR)
        # mask = mask.rotate(rotate_degree, Image.NEAREST)
        X = np.array(X)[np.newaxis, ...].astype(np.float32)  # /255
        # mask = np.array(mask)
        #
        # 1. Noise
        # c, h, w = X.shape
        # random_sigma = (np.random.rand(1) * (self.noise_sigma[1] - self.noise_sigma[0])) + self.noise_sigma[0]
        # X += np.random.randn(c, h, w) * random_sigma

        # # 2. RandomFlip
        # if self.RandomFlip:
        #     if np.random.choice([0, 1]):
        #         X = X[:, :, ::-1]
        #         mask = mask[:, ::-1]
        #
        #     if np.random.choice([0, 1]):
        #         X = X[:, ::-1, :]
        #         mask = mask[::-1, :]

        label = np.array(self.labels[index, ...])
        age = np.array(self.ages[index, ...]/75)
        sex = np.array(self.sexes[index, ...])
        return torch.from_numpy(X.astype(np.float32)), \
               torch.from_numpy(mask.astype(np.uint8)).long(), \
               torch.from_numpy(label.astype(np.uint8)).long(),\
               torch.from_numpy(age.astype(np.float32)), \
               torch.from_numpy(sex.astype(np.uint8)).long()


class cls_generator_2D_2mod(Dataset):
    def __init__(self, labeled_img1, labeled_img2, labeled_mask, labels, rotate_degree=15, noise_sigma=(0.001, 0.002)):
        self.labeled_img1 = labeled_img1
        self.labeled_img2 = labeled_img2
        self.labeled_mask = labeled_mask
        self.labels = labels
        self.degree = rotate_degree
        self.noise_sigma = noise_sigma

    def __len__(self):
        return self.labeled_img1.shape[0]

    def __getitem__(self, index):
        X1 = self.labeled_img1[index, ...]
        X2 = self.labeled_img2[index, ...]
        mask = self.labeled_mask[index, ...]
        X1 = img_utils.normalize_0_1(X1)
        X2 = img_utils.normalize_0_1(X2)
        # 0. Rotate
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        X1 = Image.fromarray(np.uint8(X1 * 255))
        X2 = Image.fromarray(np.uint8(X2 * 255))
        mask = Image.fromarray(np.uint8(mask))
        X1 = X1.rotate(rotate_degree, Image.BILINEAR)
        X2 = X2.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        X1 = np.array(X1)[np.newaxis, ...].astype(np.float32) / 255
        X2 = np.array(X2)[np.newaxis, ...].astype(np.float32) / 255
        mask = np.array(mask)

        # 1. Noise
        c, h, w = X1.shape
        random_sigma = (np.random.rand(1) * (self.noise_sigma[1] - self.noise_sigma[0])) + self.noise_sigma[0]
        X1 += np.random.randn(c, h, w) * random_sigma
        X2 += np.random.randn(c, h, w) * random_sigma
        # 2. RandomFlip
        if np.random.choice([0, 1]):
            X1 = X1[:, :, ::-1]
            X2 = X2[:, :, ::-1]
            mask = mask[:, ::-1]

        if np.random.choice([0, 1]):
            X1 = X1[:, ::-1, :]
            X2 = X2[:, ::-1, :]
            mask = mask[::-1, :]

        X = np.concatenate([X1, X2], axis=0)
        label = np.array(self.labels[index, ...])
        return torch.from_numpy(X.astype(np.float32)), \
               torch.from_numpy(mask.astype(np.uint8)).long(), \
               torch.from_numpy(label.astype(np.uint8)).long()


class cls_generator_2D_4cls(Dataset):
    def __init__(self, labeled_img, labeled_mask, labels, rotate_degree=15, noise_sigma=(0.001, 0.002)):
        self.labeled_img = labeled_img
        self.labeled_mask = labeled_mask
        self.labels = labels
        self.degree = rotate_degree
        self.noise_sigma = noise_sigma

    def __len__(self):
        return self.labeled_img.shape[0]

    def __getitem__(self, index):
        X = self.labeled_img[index, ...]
        mask = self.labeled_mask[index, ...]
        X = img_utils.normalize_0_1(X)
        # 0. Rotate
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        X = Image.fromarray(np.uint8(X * 255))
        mask = Image.fromarray(np.uint8(mask))
        X = X.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        X = np.array(X)[np.newaxis, ...].astype(np.float32) / 255
        mask = np.array(mask)

        # 1. Noise
        c, h, w = X.shape
        random_sigma = (np.random.rand(1) * (self.noise_sigma[1] - self.noise_sigma[0])) + self.noise_sigma[0]
        X += np.random.randn(c, h, w) * random_sigma

        # 2. RandomFlip
        if np.random.choice([0, 1]):
            X = X[:, :, ::-1]
            mask = mask[:, ::-1]

        if np.random.choice([0, 1]):
            X = X[:, ::-1, :]
            mask = mask[::-1, :]

        label = np.array(self.labels[index, ...])
        label_1, label_2, label_3, label_4 = np.zeros_like(label), np.zeros_like(label), \
                                             np.zeros_like(label), np.zeros_like(label)
        label_1[label == 0] = 1
        label_2[label == 1] = 1
        label_3[label == 2] = 1
        label_4[label == 3] = 1
        return torch.from_numpy(X.astype(np.float32)), \
               torch.from_numpy(mask.astype(np.uint8)).long(), \
               torch.from_numpy(label.astype(np.uint8)).long(), \
               torch.from_numpy(label_1.astype(np.uint8)).long(), \
               torch.from_numpy(label_2.astype(np.uint8)).long(), \
               torch.from_numpy(label_3.astype(np.uint8)).long(), \
               torch.from_numpy(label_4.astype(np.uint8)).long()


class data_generator_2D(Dataset):
    def __init__(self, labeled_img, labeled_mask, unlabeled):
        self.labeled_img = labeled_img
        self.labeled_mask = labeled_mask
        self.unlabeled = unlabeled

        pass

    def __len__(self):
        return {}
        pass

    def __getitem__(self, index):
        pass


def generator_2D_labeled(labeled_img, labeled_mask,
                         batch_size=1, with_tumor=False, resize=None, is_neg_sampling=False):
    labeled_number = labeled_img.shape[0]
    arr = np.arange(labeled_number)
    # print(arr)
    # np.random.shuffle(arr)
    # print(arr[:batch_size])
    while True:
        np.random.shuffle(arr)
        labeled_indices = sorted(list(arr[:batch_size]))
        X = labeled_img[labeled_indices, ...]
        X = X[:, np.newaxis, ...]
        mask = labeled_mask[labeled_indices, ...]
        if not with_tumor:
            mask[mask != 0] = 1
        label = np.sum(mask, axis=(1, 2))
        label[label != 0] = 1
        yield torch.from_numpy(X.astype(np.float32)), torch.from_numpy(mask.astype(np.uint8)).long(), \
              torch.from_numpy(label.astype(np.uint8)).long()


def generator_2D_labeled_tumor(labeled_img, labeled_mask,
                               batch_size=1, resize=None, is_neg_sampling=False):
    labeled_number = labeled_img.shape[0]
    arr = np.arange(labeled_number)
    while True:
        np.random.shuffle(arr)
        labeled_indices = sorted(list(arr[:batch_size]))
        X = labeled_img[labeled_indices, ...]
        X = X[:, np.newaxis, ...]
        mask = labeled_mask[labeled_indices, ...]
        mask[mask != 0] = 1
        label = np.sum(mask, axis=(1, 2))
        label[label != 0] = 1
        yield torch.from_numpy(X.astype(np.float32)), torch.from_numpy(mask.astype(np.uint8)).long(), \
              torch.from_numpy(label.astype(np.uint8)).long()


def generator_2D_labeled_tumor_roi(img_csv,
                                   batch_size=1, resize=None, is_neg_sampling=False):
    img_pathes = file_io.get_file_list(img_csv)
    arr = np.arange(len(img_pathes))
    while True:
        # 0. 随机获取数据索引
        np.random.shuffle(arr)
        labeled_indices = sorted(list(arr[:batch_size]))

        # 1. 读取数据


def CLS_base(root_dir, batch_size, num_class, suffix=None):
    """
    just cls ("_tumor_roi_resize_crop")
    :param root_dir:
    :param batch_size:
    :param num_class:
    :return:
    """
    with open(root_dir, 'r') as f:
        # import pdb
        # pdb.set_trace()
        strs = f.readlines()
        if suffix is None:
            img_list = [line.strip().split("|")[0] for line in strs]
        else:
            img_list = [line.strip().split("|")[0].replace("_roi_resize_crop", suffix) for line in strs]
    arr = np.arange(len(img_list) - 1)
    while True:
        np.random.shuffle(arr)
        labeled_indices = sorted(list(arr[:batch_size]))
        imgs = []
        label_out = []
        for i in labeled_indices:
            root_dir, file_name = os.path.split(img_list[i])
            _, img_arr = file_io.read_array_nii(img_list[i])  # We have transposed the data from WHD format to DHW
            img_arr = img_arr[np.newaxis, ...]
            imgs.append(img_arr)
            if num_class == 2:
                if "hemangioblastoma" in img_list[i]:
                    temp_label = 1
                else:
                    temp_label = 0

            elif num_class == 4:
                if "hemangioblastoma" in img_list[i]:  # 血管细胞瘤
                    temp_label = 0
                elif "angiocavernoma" in img_list[i]:  # 海绵状血管瘤
                    temp_label = 1
                elif "III-IV_glioma" in img_list[i]:  # 3-4型胶质瘤
                    temp_label = 2
                elif "pilocytic_astrocytoma" in img_list[i]:  # 星型细胞瘤
                    temp_label = 3
                else:
                    raise Exception("ERROR")
            label_out.append(temp_label)
        img_arrs = np.stack(imgs, axis=0)
        label_out = np.stack(label_out, axis=0)
        yield torch.from_numpy(img_arrs.astype(np.float32)), \
              torch.from_numpy(label_out.astype(np.uint8)).long()


def CLS_base_multi_task(root_dir, batch_size, num_class):
    with open(root_dir, 'r') as f:
        # import pdb
        # pdb.set_trace()
        strs = f.readlines()
        img_list = [line.strip().split("|")[0] for line in strs]
    arr = np.arange(len(img_list) - 1)
    # temp = np.eye(max(labels)+1)
    while True:
        np.random.shuffle(arr)
        labeled_indices = sorted(list(arr[:batch_size]))
        imgs = []
        masks = []
        label_out = []
        temp_label = None
        for i in labeled_indices:
            root_dir, file_name = os.path.split(img_list[i])
            _, mask_arr = file_io.read_array_nii(os.path.join(root_dir, "tumor_" + file_name))
            _, img_arr = file_io.read_array_nii(img_list[i])  # We have transposed the data from WHD format to DHW
            if num_class == 2:
                if "hemangioblastoma" in img_list[i]:
                    temp_label = 1
                else:
                    temp_label = 0
            elif num_class == 4:
                if "hemangioblastoma" in img_list[i]:  # 血管细胞瘤
                    temp_label = 0
                elif "angiocavernoma" in img_list[i]:  # 海绵状血管瘤
                    temp_label = 1
                elif "III-IV_glioma" in img_list[i]:  # 3-4型胶质瘤
                    temp_label = 2
                elif "pilocytic_astrocytoma" in img_list[i]:  # 星型细胞瘤
                    temp_label = 3
                else:
                    raise Exception("ERROR")
            mask_arr[mask_arr != 1] = 0
            img_arr = img_arr[np.newaxis, ...]
            # mask_arr = mask_arr[np.newaxis, ...]
            imgs.append(img_arr)
            masks.append(mask_arr)
            # label_out.append(temp[labels[i]])
            label_out.append(temp_label)
        try:
            img_arrs = np.stack(imgs, axis=0)
            mask_arrs = np.stack(masks, axis=0)
            label_out = np.stack(label_out, axis=0)
        except:
            print(root_dir)
        yield torch.from_numpy(img_arrs.astype(np.float32)), \
              torch.from_numpy(mask_arrs.astype(np.uint8)).long(), \
              torch.from_numpy(label_out.astype(np.uint8)).long()
        # root_dir


def CLS_base_multi_task_v2(root_dir, batch_size, num_class, is_aug=False):
    """
    version 2 adds data augmentation
    :param root_dir:
    :param batch_size:
    :param num_class:
    :return:
    """
    with open(root_dir, 'r') as f:
        # import pdb
        # pdb.set_trace()
        strs = f.readlines()
        img_list = [line.strip().split(" ")[0] for line in strs]
        mask_list = [line.strip().split(" ")[1] for line in strs]
    arr = np.arange(len(img_list) - 1)
    while True:
        np.random.shuffle(arr)
        labeled_indices = sorted(list(arr[:batch_size]))
        imgs = []
        masks = []
        label_out = []
        temp_label = None
        for i in labeled_indices:
            img = nib.load(img_list[i])
            affine = img.affine
            img = img.dataobj[np.newaxis, ...]
            mask = np.array(nib.load(mask_list[i]).dataobj)
            mask[mask != 1] = 0
            if is_aug:
                img, mask = augment.augment_data(img, mask, affine,
                                                 scale_deviation=0.2, flip=True, noise_factor=0.1,
                                                 background_correction=False, translation_deviation=0.1,
                                                 interpolation="linear")
            if num_class == 2:
                if "hemangioblastoma" in img_list[i]:
                    temp_label = 1
                else:
                    temp_label = 0
            elif num_class == 4:
                if "hemangioblastoma" in img_list[i]:  # 血管细胞瘤
                    temp_label = 0
                elif "angiocavernoma" in img_list[i]:  # 海绵状血管瘤
                    temp_label = 1
                elif "III-IV_glioma" in img_list[i]:  # 3-4型胶质瘤
                    temp_label = 2
                elif "pilocytic_astrocytoma" in img_list[i]:  # 星型细胞瘤
                    temp_label = 3
                else:
                    raise Exception("ERROR")

            # mask_arr = mask_arr[np.newaxis, ...]
            imgs.append(img)
            masks.append(mask)
            # label_out.append(temp[labels[i]])
            label_out.append(temp_label)
        try:
            img_arrs = np.stack(imgs, axis=0)
            mask_arrs = np.stack(masks, axis=0)
            label_out = np.stack(label_out, axis=0)
        except:
            raise Exception("data generate failed! ")
        yield torch.from_numpy(img_arrs.astype(np.float32)), \
              torch.from_numpy(mask_arrs.astype(np.uint8)).long(), \
              torch.from_numpy(label_out.astype(np.uint8)).long()
        # root_dir


def CLS_base_MOCO(root_dir, batch_size, num_class, is_aug=False):
    """
    version 2 adds data augmentation
    :param root_dir:
    :param batch_size:
    :param num_class:
    :return:
    """
    with open(root_dir, 'r') as f:
        # import pdb
        # pdb.set_trace()
        strs = f.readlines()
        pos_lines = []
        neg_lines = []
        for i in strs:
            if "hemangioblastoma" in i:
                pos_lines.append(i)
            else:
                neg_lines.append(i)
        pos_img_list = [line.strip().split(" ")[0] for line in pos_lines]
        pos_mask_list = [line.strip().split(" ")[1] for line in pos_lines]
        neg_img_list = [line.strip().split(" ")[0] for line in neg_lines]
        neg_mask_list = [line.strip().split(" ")[1] for line in neg_lines]
    neg_arr = np.arange(len(neg_img_list) - 1)
    pos_arr = np.arange(len(pos_img_list) - 1)
    while True:
        if np.random.randint(0, 2) == 1:
            q_label = 1
            arr = pos_arr
            img_list = pos_img_list
            mask_list = pos_mask_list
        else:
            q_label = 0
            arr = neg_arr
            img_list = neg_img_list
            mask_list = neg_mask_list
        np.random.shuffle(arr)
        labeled_indices_q = sorted(list(arr[:batch_size]))
        np.random.shuffle(arr)
        labeled_indices_k = sorted(list(arr[:batch_size]))
        # print(labeled_indices_q)
        # print(labeled_indices_k)
        imgs = []
        masks = []
        # print("data", len(labeled_indices_q+labeled_indices_k))
        for i in labeled_indices_q + labeled_indices_k:
            img = nib.load(img_list[i])
            affine = img.affine
            img = img.dataobj[:].copy()
            # import pdb
            # pdb.set_trace()
            img = img_utils.normalize_0_1(img)[np.newaxis, ...]
            mask = np.array(nib.load(mask_list[i]).dataobj)
            mask[mask != 1] = 0
            if is_aug:
                img, mask = augment.augment_data(img, mask, affine,
                                                 scale_deviation=0.1, flip=True, noise_factor=0.01,
                                                 background_correction=False, translation_deviation=0.01,
                                                 interpolation="linear")
            imgs.append(img)
            masks.append(mask)

        try:
            img_arrs = np.stack(imgs, axis=0)
            mask_arrs = np.stack(masks, axis=0)
        except:
            raise Exception("data generate failed! ")
        yield [torch.from_numpy(img_arrs[:batch_size, ...].astype(np.float32)),
               torch.from_numpy(mask_arrs[:batch_size, ...].astype(np.uint8)).long()], \
              [torch.from_numpy(img_arrs[batch_size:, ...].astype(np.float32)),
               torch.from_numpy(mask_arrs[batch_size:, ...].astype(np.uint8)).long()], \
              torch.tensor(q_label).long()
        # root_dir


class CLS_base_MOCO_train(Dataset):
    def __init__(self, root_dir):
        with open(root_dir, 'r') as f:
            # import pdb
            # pdb.set_trace()
            strs = f.readlines()
            label = []
            imgs = [line.strip().split(" ")[0] for line in strs]
            # print(imgs)
            for i in imgs:
                if "hemangioblastoma" in i:
                    label.append(1)
                else:
                    label.append(0)
        self.img_list = imgs
        self.labels = label
            # self.pos_mask_list = [line.strip().split(" ")[1] for line in pos_lines]
            # self.neg_img_list = [line.strip().split(" ")[0] for line in neg_lines]
            # self.neg_mask_list = [line.strip().split(" ")[1] for line in neg_lines]
        # self.neg_arr = np.arange(len(self.neg_img_list) - 1)
        # self.pos_arr = np.arange(len(self.pos_img_list) - 1)
        print(self.img_list[0])
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # print(idx)
        img = nib.load(self.img_list[idx])
        affine = img.affine
        img = img.dataobj[:].copy()
        mask = np.zeros_like(img)
        # print(self.img_list[idx])
        # print(img.shape)
        img = img_utils.normalize_0_1(img)
        img, _ = augment.augment_data(img[np.newaxis, ...], mask, affine,
                                         scale_deviation=0.1, flip=True, noise_factor=0.05,
                                         background_correction=False, translation_deviation=0.01,
                                         interpolation="linear")

        # import pdb
        # pdb.set_trace()
        # img = img.squeeze()
        return torch.from_numpy(img.astype(np.float32)), torch.tensor(self.labels[idx]).long()


class CLS_base_MOCO_train_seg(Dataset):
    def __init__(self, root_dir):
        self.mask_list = []
        self.img_list = []
        with open(root_dir, 'r') as f:
            # import pdb
            # pdb.set_trace()
            strs = f.readlines()
            label = []
            self.img_list = [line.strip().split(" ")[0] for line in strs]
            self.mask_list = [line.strip().split(" ")[1] for line in strs]
            # print(self.img_list)
            for i in self.img_list:
                if "hemangioblastoma" in i:
                    label.append(1)
                else:
                    label.append(0)
        self.labels = label
        print(self.img_list[0])
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # print(idx)
        # print(self.img_list[idx])
        mask = np.array(nib.load(self.mask_list[idx]).dataobj)
        img = nib.load(self.img_list[idx])
        affine = img.affine
        img = img.dataobj[:].copy()
        img = img_utils.normalize_0_1(img)[np.newaxis, ...]
        mask[mask != 1] = 0
        img, mask = augment.augment_data(img, mask, affine,scale_deviation=0.1, flip=True,
                                         noise_factor=0.05,background_correction=False,
                                         translation_deviation=0.01,interpolation="linear")

        # import pdb
        # pdb.set_trace()
        # img = img.squeeze()
        return torch.from_numpy(img.astype(np.float32)), \
               torch.from_numpy(mask.astype(np.uint8)).long(),\
               torch.tensor(self.labels[idx]).long()



class CLS_base_MOCO_train_seg_v2(Dataset):
    def __init__(self, root_dir, is_aug=True):
        self.mask_list = []
        self.img_list = []
        with open(root_dir, 'r') as f:
            # import pdb
            # pdb.set_trace()
            strs = f.readlines()
            label = []
            self.img_list = [line.strip().split(" ")[0] for line in strs]
            self.mask_list = [line.strip().split(" ")[1] for line in strs]
            # print(self.img_list)
            for i in self.img_list:
                if "hemangioblastoma" in i:
                    label.append(1)
                else:
                    label.append(0)
        self.labels = label
        print(self.img_list[0])
        data = open("./data/name_age/age_dataset.json", "r")
        self.age_dataset = json.load(data)
        data.close()
        data = open("./data/name_age/sex_dataset.json", "r")
        self.sex_dataset = json.load(data)
        data.close()
        self.is_aug = is_aug
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # print(idx)
        # print(self.img_list[idx])
        try:
            tumor, name = self.mask_list[idx].split('/')[8:10]
            age = self.age_dataset['_'.join([tumor,name])]
            age = torch.tensor(age/100)
            sex = torch.tensor(self.sex_dataset['_'.join([tumor,name])])
        except:
            tumor, name = self.mask_list[idx].split('/')[8:10]
            print(tumor, name)
            age = torch.tensor(0)
            sex = torch.tensor(0)

        mask = np.array(nib.load(self.mask_list[idx].replace('/data/data_disk/zhaobt/', '../../')).dataobj)
        img = nib.load(self.img_list[idx].replace('/data/data_disk/zhaobt/', '../../'))
        affine = img.affine
        img = img.dataobj[:].copy()
        img = img_utils.normalize_0_1(img)[np.newaxis, ...]
        mask[mask != 1] = 0
        if self.is_aug:
            img, mask = augment.augment_data(img, mask, affine,scale_deviation=0.1, flip=True,
                                             noise_factor=0.05,background_correction=False,
                                             translation_deviation=0.01,interpolation="linear")

        # import pdb
        # pdb.set_trace()
        # img = img.squeeze()
        return torch.from_numpy(img.astype(np.float32)), \
               torch.from_numpy(mask.astype(np.uint8)).long(),\
               torch.tensor(self.labels[idx]).long(), age, sex


class CLS_base_MOCO_train_seg_v3(Dataset):
    def __init__(self, root_dir, is_aug=True):
        self.mask_list = []
        self.img_list = []
        with open(root_dir, 'r') as f:
            # import pdb
            # pdb.set_trace()
            strs = f.readlines()
            label = []
            self.img_list = [line.strip().split(" ")[0] for line in strs]
            self.mask_list = [line.strip().split(" ")[1] for line in strs]
            # print(self.img_list)
            for i in self.img_list:
                if "hemangioblastoma" in i:
                    label.append(1)
                else:
                    label.append(0)
        self.labels = label
        print(self.img_list[0])
        data = open("./data/name_age/age_dataset.json", "r")
        self.age_dataset = json.load(data)
        data.close()
        data = open("./data/name_age/sex_dataset.json", "r")
        self.sex_dataset = json.load(data)
        data.close()
        self.is_aug = is_aug
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # print(idx)
        # print(self.img_list[idx])
        try:
            tumor, name = self.mask_list[idx].split('/')[8:10]
            age = self.age_dataset['_'.join([tumor,name])]
            age = torch.tensor(age/100)
            sex = torch.tensor(self.sex_dataset['_'.join([tumor,name])])
        except:
            tumor, name = self.mask_list[idx].split('/')[8:10]
            print(tumor, name)
            age = torch.tensor(0)
            sex = torch.tensor(0)

        mask = np.array(nib.load(self.mask_list[idx].replace('/data/data_disk/zhaobt/', '../../')).dataobj)
        img = nib.load(self.img_list[idx].replace('/data/data_disk/zhaobt/', '../../'))
        affine = img.affine
        img = img.dataobj[:].copy()
        img = np.pad(img, ((0, 0), (8, 8), (0, 0)), 'constant', constant_values=(0, 0))
        img = img_utils.normalize_0_1(img)[np.newaxis, ...]
        mask[mask != 1] = 0
        # np.pad(A, ((3, 2), (2, 3)), 'constant', constant_values=(0, 0))

        mask = np.pad(mask, ((0, 0), (8, 8), (0, 0)), 'constant', constant_values=(0, 0))
        if self.is_aug:
            img, mask = augment.augment_data(img, mask, affine,scale_deviation=0.1, flip=True,
                                             noise_factor=0.05,background_correction=False,
                                             translation_deviation=0.01,interpolation="linear")

        # import pdb
        # pdb.set_trace()
        # img = img.squeeze()
        return torch.from_numpy(img.astype(np.float32)), \
               torch.from_numpy(mask.astype(np.uint8)).long(),\
               torch.tensor(self.labels[idx]).long(), age, sex



class moco_sampler(Sampler):
    """
    由于每个batch的点数可能不一致
    例如 len(b[0])=10220, len(b[1])=23300, len(b[2])=24000 , ...
    该sampler是为了将每个batch内的点数统一
    首先将batch里的样本按照点数从小到大排列
    返回排序之后的索引值
    """
    def __init__(self, dataset, batch_size, iters):
        # super(moco_sampler, self).__init__()
        self.x = dataset
        self.batch_size = batch_size
        self.iters = iters
        # print(self.x)
        self.pos_id = []
        self.neg_id = []
        for idx, d in enumerate(self.x.labels):
            if d == 1:
                self.pos_id.append(int(idx))
            else:
                self.neg_id.append(int(idx))

    def __iter__(self):
        idx = 0
        while True:
            if idx>= self.iters:
                break
            if np.random.randint(0, 2) == 1:
                temp_list = self.pos_id
            else:
                temp_list = self.neg_id
            np.random.shuffle(temp_list)
            labeled_indices_q = list(temp_list[:self.batch_size//2])
            np.random.shuffle(temp_list)
            labeled_indices_k = list(temp_list[:self.batch_size//2])
            batch = labeled_indices_q+labeled_indices_k
            idx += 1
            yield batch
        # if len(batch) == self.batch_size:
        #     print('=============================')
        #     yield from batch
        #     batch = []
        # if len(batch) > 0:
        #     yield from batch

    # def __len__(self):
    #     return len(self.x[0])  # 这里的__len__需要返回总长度


class moco_sampler2(Sampler):
    """
    由于每个batch的点数可能不一致
    例如 len(b[0])=10220, len(b[1])=23300, len(b[2])=24000 , ...
    该sampler是为了将每个batch内的点数统一
    首先将batch里的样本按照点数从小到大排列
    返回排序之后的索引值
    """
    def __init__(self, dataset, batch_size, iters):
        # super(moco_sampler, self).__init__()
        self.x = dataset
        self.batch_size = batch_size
        self.iters = iters
        # print(self.x)
        self.pos_id = []
        self.neg_id = []
        for idx, d in enumerate(self.x.labels):
            if d == 1:
                self.pos_id.append(int(idx))
            else:
                self.neg_id.append(int(idx))
        self.all_data = self.pos_id+self.neg_id
    def __iter__(self):
        idx = 0
        while True:
            if idx>= self.iters:
                break
            # if np.random.randint(0, 2) == 1:
            #     temp_list = self.pos_id
            # else:
            #     temp_list = self.neg_id
            temp_list = self.all_data
            np.random.shuffle(temp_list)
            batch = temp_list[:self.batch_size]
            idx += 1
            yield batch
        # if len(batch) == self.batch_size:
        #     print('=============================')
        #     yield from batch
        #     batch = []
        # if len(batch) > 0:
        #     yield from batch

    # def __len__(self):
    #     return len(self.x[0])  # 这里的__len__需要返回总长度



def collate_fn(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    print(data[0].shape)
    return [torch.stack(data), torch.tensor(labels)]
# def collate_fn(arr, img_list, mask_list):
#     np.random.shuffle(arr)
#     labeled_indices_q = sorted(list(arr[:batch_size]))
#     np.random.shuffle(arr)
#     labeled_indices_k = sorted(list(arr[:batch_size]))
#     # print(labeled_indices_q)
#     # print(labeled_indices_k)
#     imgs = []
#     masks = []
#     # print("data", len(labeled_indices_q+labeled_indices_k))
#     for i in labeled_indices_q + labeled_indices_k:
#         img = nib.load(img_list[i])
#         affine = img.affine
#         img = img.dataobj[:].copy()
#         # import pdb
#         # pdb.set_trace()
#         img = img_utils.normalize_0_1(img)[np.newaxis, ...]
#         mask = np.array(nib.load(mask_list[i]).dataobj)
#         mask[mask != 1] = 0
#         if is_aug:
#             img, mask = augment.augment_data(img, mask, affine,
#                                              scale_deviation=0.1, flip=True, noise_factor=0.01,
#                                              background_correction=False, translation_deviation=0.01,
#                                              interpolation="linear")
#         imgs.append(img)
#         masks.append(mask)
#
#     try:
#         img_arrs = np.stack(imgs, axis=0)
#         mask_arrs = np.stack(masks, axis=0)
#     except:
#         raise Exception("data generate failed! ")
#     yield [torch.from_numpy(img_arrs[:batch_size, ...].astype(np.float32)),
#            torch.from_numpy(mask_arrs[:batch_size, ...].astype(np.uint8)).long()], \
#           [torch.from_numpy(img_arrs[batch_size:, ...].astype(np.float32)),
#            torch.from_numpy(mask_arrs[batch_size:, ...].astype(np.uint8)).long()], \
#           torch.tensor(q_label).long()



class CLS_base_MOCO_pos(Dataset):
    def __init__(self, rootdir):
        with open(rootdir, 'r') as f:
            # import pdb
            # pdb.set_trace()
            strs = f.readlines()
            pos_lines = []
            neg_lines = []
            for i in strs:
                if "hemangioblastoma" in i:
                    pos_lines.append(i)
                else:
                    neg_lines.append(i)
            self.pos_img_list = [line.strip().split(" ")[0] for line in pos_lines] * 50
            self.pos_mask_list = [line.strip().split(" ")[1] for line in pos_lines] * 50

    def __len__(self):
        return len(self.pos_img_list)

    def __getitem__(self, idx):
        img = nib.load(self.pos_img_list[idx].replace("zhang_istbi", "zhang_istbi/data_disk"))
        affine = img.affine
        img = img.dataobj[:].copy()
        # import pdb
        # pdb.set_trace()
        img = img_utils.normalize_0_1(img)[np.newaxis, ...]
        mask = np.array(nib.load(self.pos_mask_list[idx].replace("zhang_istbi", "zhang_istbi/data_disk")).dataobj)
        mask[mask != 1] = 0
        img, mask = augment.augment_data(img, mask, affine,
                                         scale_deviation=0.1, flip=True, noise_factor=0.01,
                                         background_correction=False, translation_deviation=0.01,
                                         interpolation="linear")
        return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(mask.astype(np.uint8)).long()


class CLS_base_MOCO_neg(Dataset):
    def __init__(self, rootdir):
        with open(rootdir, 'r') as f:
            # import pdb
            # pdb.set_trace()
            strs = f.readlines()
            pos_lines = []
            neg_lines = []
            for i in strs:
                if "hemangioblastoma" in i:
                    pos_lines.append(i)
                else:
                    neg_lines.append(i)
            self.neg_img_list = [line.strip().split(" ")[0] for line in neg_lines] * 50
            self.neg_mask_list = [line.strip().split(" ")[1] for line in neg_lines] * 50

    def __len__(self):
        return len(self.neg_img_list)

    def __getitem__(self, idx):
        img = nib.load(self.neg_img_list[idx].replace("zhang_istbi", "zhang_istbi/data_disk"))
        affine = img.affine
        img = img.dataobj[:].copy()
        # import pdb
        # pdb.set_trace()
        img = img_utils.normalize_0_1(img)[np.newaxis, ...]
        mask = np.array(nib.load(self.neg_mask_list[idx].replace("zhang_istbi", "zhang_istbi/data_disk")).dataobj)
        mask[mask != 1] = 0
        img, mask = augment.augment_data(img, mask, affine,
                                         scale_deviation=0.1, flip=True, noise_factor=0.01,
                                         background_correction=False, translation_deviation=0.01,
                                         interpolation="linear")
        return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(mask.astype(np.uint8)).long()


def CLS_by_seg(root_dir, batch_size, num_class, is_aug=False):
    """
    通过分割的方式分类： 类比T1-net
    :param root_dir:
    :param batch_size:
    :param num_class:
    :return:
    """
    with open(root_dir, 'r') as f:
        # import pdb
        # pdb.set_trace()
        strs = f.readlines()
        img_list = [line.strip().split(" ")[0] for line in strs]
        mask_list = [line.strip().split(" ")[1] for line in strs]
    arr = np.arange(len(img_list) - 1)
    while True:
        np.random.shuffle(arr)
        labeled_indices = sorted(list(arr[:batch_size]))
        imgs = []
        masks = []
        label_out = []
        temp_label = None
        for i in labeled_indices:
            img = nib.load(img_list[i])
            affine = img.affine
            img = img.dataobj[:].copy()
            # import pdb
            # pdb.set_trace()
            img = img_utils.normalize_0_1(img)[np.newaxis, ...]
            mask = np.array(nib.load(mask_list[i]).dataobj)
            mask[mask != 1] = 0

            if is_aug:
                img, mask = augment.augment_data(img, mask, affine, scale_deviation=0.05, flip=True, noise_factor=0.01,
                                                 background_correction=False, translation_deviation=0.01,
                                                 interpolation="linear")
            if num_class == 2:
                if "hemangioblastoma" in img_list[i]:
                    mask = mask * 1
                else:
                    mask = mask * 2
            elif num_class == 4:
                # if "hemangioblastoma" in img_list[i]:  # 血管细胞瘤
                #     temp_label = 0
                # elif "angiocavernoma" in img_list[i]:  # 海绵状血管瘤
                #     temp_label = 1
                # elif "III-IV_glioma" in img_list[i]:  # 3-4型胶质瘤
                #     temp_label = 2
                # elif "pilocytic_astrocytoma" in img_list[i]:  # 星型细胞瘤
                #     temp_label = 3
                # else:
                raise Exception("Not implemented")

            imgs.append(img)
            masks.append(mask)
            label_out.append(temp_label)
        try:
            img_arrs = np.stack(imgs, axis=0)
            mask_arrs = np.stack(masks, axis=0)
            label_out = np.stack(label_out, axis=0)
        except:
            raise Exception("data generate failed! ")
        yield torch.from_numpy(img_arrs.astype(np.float32)), \
              torch.from_numpy(mask_arrs.astype(np.uint8)).long()


class VAL_CLS_by_seg(Dataset):
    def __init__(self, root_dir, out_dir, num_class):
        with open(root_dir, 'r') as f:
            # import pdb
            # pdb.set_trace()
            strs = f.readlines()
            print("val data number:", len(strs))
            self.img_list = [line.strip().split(" ")[0] for line in strs]
            self.mask_list = [line.strip().split(" ")[1] for line in strs]
            # self.labels = [int(line.strip().split('|')[1]) for line in strs]
            self.labels = []
            for idx, i in enumerate(self.img_list):
                if num_class == 2:
                    if "hemangioblastoma" in i:
                        temp_label = 1
                    else:
                        temp_label = 0
                else:
                    raise Exception("Not implemented!")
                self.labels.append(temp_label)
        self.outdir = out_dir
        print("Processing {} datas, {} labels".format(len(self.img_list), len(self.labels)))
        print(self.labels)
        print(self.img_list[0])
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # read image and labels
        img_name = self.img_list[idx]
        img_name_block = img_name.split("/")
        out_path = os.path.join(self.outdir, "-".join(img_name_block[7:9]))
        label_n = np.array(self.labels[idx])
        assert os.path.isfile(img_name)
        _, img_arr = file_io.read_array_nii(img_name)  # We have transposed the data from WHD format to DHW
        _, mask_arr = file_io.read_array_nii(self.mask_list[idx])
        img_arr = img_utils.normalize_0_1(img_arr)
        assert img_arr is not None
        assert mask_arr is not None
        mask_arr[mask_arr != 1] = 0
        # mask_arr = mask_arr
        img_arr = img_arr[np.newaxis, ...]
        return torch.from_numpy(img_arr.astype(np.float32)), \
               torch.from_numpy(mask_arr.astype(np.uint8)).long(),\
               torch.from_numpy(label_n.astype(np.uint8)).long(), out_path\

class VAL_CLS_by_seg_v2(Dataset):
    def __init__(self, root_dir, out_dir, num_class):
        with open(root_dir, 'r') as f:
            # import pdb
            # pdb.set_trace()
            strs = f.readlines()
            print("val data number:", len(strs))
            self.img_list = [line.strip().split(" ")[0] for line in strs]
            self.mask_list = [line.strip().split(" ")[1] for line in strs]
            # self.labels = [int(line.strip().split('|')[1]) for line in strs]
            self.labels = []
            for idx, i in enumerate(self.img_list):
                if num_class == 2:
                    if "hemangioblastoma" in i:
                        temp_label = 1
                    else:
                        temp_label = 0
                else:
                    raise Exception("Not implemented!")
                self.labels.append(temp_label)
        self.outdir = out_dir
        print("Processing {} datas, {} labels".format(len(self.img_list), len(self.labels)))
        # print(self.labels)
        # print(self.img_list[0])
        data = open("./data/name_age/age_dataset.json", "r")
        self.age_dataset = json.load(data)
        data.close()
        data = open("./data/name_age/sex_dataset.json", "r")
        self.sex_dataset = json.load(data)
        data.close()
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # read image and labels
        img_name = self.img_list[idx]
        tumor, name = self.mask_list[idx].split('/')[8:10]
        img_name_block = img_name.split("/")
        out_path = os.path.join(self.outdir, "-".join(img_name_block[7:9]))
        try:
            age = self.age_dataset['_'.join([tumor,name])]
            age = torch.tensor(age/100)
            sex = torch.tensor(self.sex_dataset['_'.join([tumor,name])])
        except:
            age, sex = 0, 0
            print('_'.join([tumor,name]))
        label_n = np.array(self.labels[idx])
        # assert os.path.isfile(img_name)
        _, img_arr = file_io.read_array_nii(img_name.replace('/data/data_disk/zhaobt/', '../../'))  # We have transposed the data from WHD format to DHW
        _, mask_arr = file_io.read_array_nii(self.mask_list[idx].replace('/data/data_disk/zhaobt/', '../../'))
        img_arr = img_utils.normalize_0_1(img_arr)
        assert img_arr is not None
        assert mask_arr is not None
        mask_arr[mask_arr != 1] = 0
        # mask_arr = mask_arr
        img_arr = img_arr[np.newaxis, ...]
        return torch.from_numpy(img_arr.astype(np.float32)), \
               torch.from_numpy(mask_arr.astype(np.uint8)).long(),\
               torch.from_numpy(label_n.astype(np.uint8)).long(), age, sex, out_path

class VAL_CLS_by_seg_v3(Dataset):
    def __init__(self, root_dir, out_dir, num_class):
        with open(root_dir, 'r') as f:
            # import pdb
            # pdb.set_trace()
            strs = f.readlines()
            print("val data number:", len(strs))
            self.img_list = [line.strip().split(" ")[0] for line in strs]
            self.mask_list = [line.strip().split(" ")[1] for line in strs]
            # self.labels = [int(line.strip().split('|')[1]) for line in strs]
            self.labels = []
            for idx, i in enumerate(self.img_list):
                if num_class == 2:
                    if "hemangioblastoma" in i:
                        temp_label = 1
                    else:
                        temp_label = 0
                else:
                    raise Exception("Not implemented!")
                self.labels.append(temp_label)
        self.outdir = out_dir
        print("Processing {} datas, {} labels".format(len(self.img_list), len(self.labels)))
        # print(self.labels)
        # print(self.img_list[0])
        data = open("./data/name_age/age_dataset.json", "r")
        self.age_dataset = json.load(data)
        data.close()
        data = open("./data/name_age/sex_dataset.json", "r")
        self.sex_dataset = json.load(data)
        data.close()
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # read image and labels
        img_name = self.img_list[idx]
        tumor, name = self.mask_list[idx].split('/')[8:10]
        img_name_block = img_name.split("/")
        out_path = os.path.join(self.outdir, "-".join(img_name_block[7:9]))
        try:
            age = self.age_dataset['_'.join([tumor,name])]
            age = torch.tensor(age/100)
            sex = torch.tensor(self.sex_dataset['_'.join([tumor,name])])
        except:
            age, sex = 0, 0
            print('_'.join([tumor,name]))
        label_n = np.array(self.labels[idx])
        # assert os.path.isfile(img_name)
        _, img_arr = file_io.read_array_nii(img_name.replace('/data/data_disk/zhaobt/', '../../'))  # We have transposed the data from WHD format to DHW
        _, mask_arr = file_io.read_array_nii(self.mask_list[idx].replace('/data/data_disk/zhaobt/', '../../'))
        img_arr = img_utils.normalize_0_1(img_arr)
        assert img_arr is not None
        assert mask_arr is not None
        mask_arr[mask_arr != 1] = 0
        img_arr = np.pad(img_arr, ((0,0), (8,8), (0,0)), 'constant', constant_values=(0, 0))
        mask_arr = np.pad(mask_arr, ((0, 0), (8, 8), (0, 0)), 'constant', constant_values=(0, 0))
        # mask_arr = mask_arr
        img_arr = img_arr[np.newaxis, ...]
        return torch.from_numpy(img_arr.astype(np.float32)), \
               torch.from_numpy(mask_arr.astype(np.uint8)).long(),\
               torch.from_numpy(label_n.astype(np.uint8)).long(), age, sex, out_path



class VAL_CLS_moco(Dataset):
    def __init__(self, train_dataset, outdir, num_class):
        self.outdir = outdir
        with open(train_dataset, 'r') as f:
            # import pdb
            # pdb.set_trace()
            strs = f.readlines()
            print("val data number:", len(strs))
            self.img_list = [line.strip().split(" ")[0] for line in strs]
            self.mask_list = [line.strip().split(" ")[1] for line in strs]
            self.labels = []
            for idx, i in enumerate(self.img_list):
                if "hemangioblastoma" in i:
                    q_label = 1
                else:
                    q_label = 0
                self.labels.append(q_label)
        print("Processing {} datas, {} labels".format(len(self.img_list), len(self.labels)))
        print(self.labels)
        print(self.img_list[0])
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # read image and labels

        img_name = self.img_list[idx]

        img_name_block = img_name.split("/")
        out_path = os.path.join(self.outdir, "-".join(img_name_block[7:9]))
        label_n = np.array(self.labels[idx])
        _, img_arr = file_io.read_array_nii(img_name)  # We have transposed the data from WHD format to DHW
        img_arr = img_utils.normalize_0_1(img_arr)
        _, mask_arr = file_io.read_array_nii(self.mask_list[idx])
        assert img_arr is not None
        assert mask_arr is not None
        mask_arr[mask_arr != 1] = 0
        img_arr = img_arr[np.newaxis, ...]
        return torch.from_numpy(img_arr.astype(np.float32)), \
               torch.from_numpy(mask_arr.astype(np.uint8)).long(), \
               torch.from_numpy(label_n.astype(np.uint8)).long(), \
               out_path

class VAL_CLS_moco_vae(Dataset):
    def __init__(self, train_dataset, outdir, num_class):
        self.outdir = outdir
        with open(train_dataset, 'r') as f:
            # import pdb
            # pdb.set_trace()
            strs = f.readlines()
            print("val data number:", len(strs))
            self.img_list = [line.strip().split(" ")[0] for line in strs]
            self.mask_list = [line.strip().split(" ")[1] for line in strs]
            self.labels = []
            for idx, i in enumerate(self.img_list):
                if "hemangioblastoma" in i:
                    q_label = 1
                else:
                    q_label = 0
                self.labels.append(q_label)
        print("Processing {} datas, {} labels".format(len(self.img_list), len(self.labels)))
        print(self.labels)
        print(self.img_list[0])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # read image and labels

        img_name = self.img_list[idx]
        img_name_block = img_name.split("/")
        out_path = os.path.join(self.outdir, "-".join(img_name_block[7:9]))
        label_n = np.array(self.labels[idx])
        img_arr = nib.load(img_name)
        img_arr = img_arr.dataobj[:].copy()
        img_arr = img_utils.normalize_0_1(img_arr)
        assert img_arr is not None
        img_arr = img_arr[np.newaxis, ...]
        return torch.from_numpy(img_arr.astype(np.float32)), \
               torch.from_numpy(label_n.astype(np.uint8)).long(), \
               out_path


# def CLS_base_multi_task(root_dir, batch_size, num_class, is_aug=False):
#     """
#     version 2 adds data augmentation
#     :param root_dir:
#     :param batch_size:
#     :param num_class:
#     :return:
#     """
#     with open(root_dir, 'r') as f:
#         # import pdb
#         # pdb.set_trace()
#         strs = f.readlines()
#         img_list = [line.strip().split(" ")[0] for line in strs]
#         mask_list = [line.strip().split(" ")[1] for line in strs]
#     arr = np.arange(len(img_list)-1)
#     while True:
#         np.random.shuffle(arr)
#         labeled_indices = sorted(list(arr[:batch_size]))
#         imgs = []
#         masks = []
#         label_out = []
#         temp_label = None
#         for i in labeled_indices:
#             img = nib.load(img_list[i])
#             affine = img.affine
#             img = img.dataobj[np.newaxis, ...]
#             mask = np.array(nib.load(mask_list[i]).dataobj)
#             mask[mask != 1] = 0
#             if is_aug:
#                 img, mask = augment.augment_data(img, mask, affine,
#                                                  scale_deviation=0.2, flip=True, noise_factor=0.1,
#                                                  background_correction=False, translation_deviation=0.1,
#                                                  interpolation="linear")
#             if num_class == 2:
#                 if "hemangioblastoma" in img_list[i]:
#                     temp_label = 1
#                 else:
#                     temp_label = 0
#             elif num_class == 4:
#                 if "hemangioblastoma" in img_list[i]:  # 血管细胞瘤
#                     temp_label = 0
#                 elif "angiocavernoma" in img_list[i]:  # 海绵状血管瘤
#                     temp_label = 1
#                 elif "III-IV_glioma" in img_list[i]:  # 3-4型胶质瘤
#                     temp_label = 2
#                 elif "pilocytic_astrocytoma" in img_list[i]:  # 星型细胞瘤
#                     temp_label = 3
#                 else:
#                     raise Exception("ERROR")
#
#
#             # mask_arr = mask_arr[np.newaxis, ...]
#             imgs.append(img)
#             masks.append(mask)
#             # label_out.append(temp[labels[i]])
#             label_out.append(temp_label)
#         try:
#             img_arrs = np.stack(imgs, axis=0)
#             mask_arrs = np.stack(masks, axis=0)
#             label_out = np.stack(label_out, axis=0)
#         except:
#             raise Exception("data generate failed! ")
#         yield torch.from_numpy(img_arrs.astype(np.float32)), \
#               torch.from_numpy(mask_arrs.astype(np.uint8)).long(), \
#               torch.from_numpy(label_out.astype(np.uint8)).long()
#               # root_dir


class Cls_base_test(Dataset):
    def __init__(self, root_dir, num_class, suffix=None, return_name=False):
        self.return_name = return_name
        with open(root_dir, 'r') as f:
            strs = f.readlines()
            if suffix is None:
                self.img_list = [line.strip().split("|")[0] for line in strs]
            else:
                self.img_list = [line.strip().split("|")[0].replace("_roi_resize_crop", suffix) for line in strs]

            self.labels = []
            for idx, i in enumerate(self.img_list):
                if num_class == 2:
                    if "hemangioblastoma" in i:
                        temp_label = 1
                    else:
                        temp_label = 0
                elif num_class == 4:
                    if "hemangioblastoma" in i:  # 血管细胞瘤
                        temp_label = 0
                    elif "angiocavernoma" in i:  # 海绵状血管瘤
                        temp_label = 1
                    elif "III-IV_glioma" in i:  # 3-4型胶质瘤
                        temp_label = 2
                    elif "pilocytic_astrocytoma" in i:  # 星型细胞瘤
                        temp_label = 3
                    else:
                        raise Exception("ERROR")
                self.labels.append(temp_label)
        self.num_class = num_class
        print("Processing {} datas, {} labels".format(len(self.img_list), len(self.labels)))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # read image and labels
        img_name = self.img_list[idx]
        label_n = np.array(self.labels[idx])
        assert os.path.isfile(img_name)
        _, img_arr = file_io.read_array_nii(img_name)  # We have transposed the data from WHD format to DHW
        # print(img_name)
        assert img_arr is not None
        img_arr = img_arr[np.newaxis, ...]
        if self.return_name:
            return torch.from_numpy(img_arr.astype(np.float32)), \
                   torch.from_numpy(label_n.astype(np.uint8)).long(), \
                   img_name
        return torch.from_numpy(img_arr.astype(np.float32)), \
               torch.from_numpy(label_n.astype(np.uint8)).long()


class VAL_CLS_base_multi_task(Dataset):
    def __init__(self, root_dir, num_class):
        with open(root_dir, 'r') as f:
            # import pdb
            # pdb.set_trace()
            strs = f.readlines()
            print("val data number:", len(strs))
            self.img_list = [line.strip().split(" ")[0] for line in strs]
            self.mask_list = [line.strip().split(" ")[1] for line in strs]
            # self.labels = [int(line.strip().split('|')[1]) for line in strs]
            self.labels = []
            for idx, i in enumerate(self.img_list):
                if num_class == 2:
                    if "hemangioblastoma" in i:
                        temp_label = 1
                    else:
                        temp_label = 0
                elif num_class == 4:
                    if "hemangioblastoma" in i:  # 血管细胞瘤
                        temp_label = 0
                    elif "angiocavernoma" in i:  # 海绵状血管瘤
                        temp_label = 1
                    elif "III-IV_glioma" in i:  # 3-4型胶质瘤
                        temp_label = 2
                    elif "pilocytic_astrocytoma" in i:  # 星型细胞瘤
                        temp_label = 3
                    else:
                        raise Exception("ERROR")
                self.labels.append(temp_label)
        print("Processing {} datas, {} labels".format(len(self.img_list), len(self.labels)))
        print(self.labels)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # read image and labels

        img_name = self.img_list[idx]
        label_n = np.array(self.labels[idx])
        assert os.path.isfile(img_name)
        _, img_arr = file_io.read_array_nii(img_name)  # We have transposed the data from WHD format to DHW
        # print(img_name)

        _, mask_arr = file_io.read_array_nii(self.mask_list[idx])
        assert img_arr is not None
        assert mask_arr is not None
        mask_arr[mask_arr != 1] = 0
        img_arr = img_arr[np.newaxis, ...]
        return torch.from_numpy(img_arr.astype(np.float32)), \
               torch.from_numpy(mask_arr.astype(np.uint8)).long(), \
               torch.from_numpy(label_n.astype(np.uint8)).long()


class Cls_base_multi_task(Dataset):
    def __init__(self, root_dir, num_class):
        with open(root_dir, 'r') as f:
            # import pdb
            # pdb.set_trace()
            strs = f.readlines()
            self.img_list = [line.strip().split("|")[0] for line in strs]
            # self.labels = [int(line.strip().split('|')[1]) for line in strs]
            self.labels = []
            for idx, i in enumerate(self.img_list):
                if num_class == 2:
                    if "hemangioblastoma" in i:
                        temp_label = 1
                    else:
                        temp_label = 0
                elif num_class == 4:
                    if "hemangioblastoma" in i:  # 血管细胞瘤
                        temp_label = 0
                    elif "angiocavernoma" in i:  # 海绵状血管瘤
                        temp_label = 1
                    elif "III-IV_glioma" in i:  # 3-4型胶质瘤
                        temp_label = 2
                    elif "pilocytic_astrocytoma" in i:  # 星型细胞瘤
                        temp_label = 3
                    else:
                        raise Exception("ERROR")
                self.labels.append(temp_label)
        print("Processing {} datas, {} labels".format(len(self.img_list), len(self.labels)))
        print(self.labels)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # read image and labels

        img_name = self.img_list[idx]
        label_n = np.array(self.labels[idx])
        assert os.path.isfile(img_name)
        _, img_arr = file_io.read_array_nii(img_name)  # We have transposed the data from WHD format to DHW
        # print(img_name)

        root_dir, file_name = os.path.split(self.img_list[idx])
        _, mask_arr = file_io.read_array_nii(os.path.join(root_dir, "tumor_" + file_name))
        assert img_arr is not None
        assert mask_arr is not None
        mask_arr[mask_arr != 1] = 0
        img_arr = img_arr[np.newaxis, ...]
        return torch.from_numpy(img_arr.astype(np.float32)), \
               torch.from_numpy(mask_arr.astype(np.uint8)).long(), \
               torch.from_numpy(label_n.astype(np.uint8)).long()


class generator_3D_roi(Dataset):
    def __init__(self, img_csv, with_tumor=False):
        self.img_pathes = file_io.get_file_list(img_csv)

    def __len__(self):
        return len(self.img_pathes)

    def __getitem__(self, item):
        img_dir = self.img_pathes[item].replace(".nii.gz", "_roi.nii.gz")
        mask_dir = self.img_pathes[item].replace(".nii.gz", "_roi.nii.gz")
        img, img_arr = file_io.read_array_nii(img_dir)

        X = self.img[item, ...]
        X = X[np.newaxis, ...]
        mask = self.mask[item, ...]
        if not self.with_tumor:
            mask[mask != 0] = 1
        label = np.array([1]) if np.sum(mask) != 0 else np.array([0])
        return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(mask.astype(np.uint8)).long(), \
               torch.from_numpy(label.astype(np.uint8)).long()


class generator_2D_val(Dataset):
    def __init__(self, labeled_img, labeled_mask, with_tumor=False):
        self.img = labeled_img
        self.mask = labeled_mask
        self.with_tumor = with_tumor

    def __len__(self):
        return self.mask.shape[0]

    def __getitem__(self, item):
        X = self.img[item, ...]
        X = X[np.newaxis, ...]
        mask = self.mask[item, ...]
        # if not self.with_tumor:
        #     mask[mask != 0] = 1
        label = np.array([1]) if np.sum(mask) != 0 else np.array([0])
        return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(mask.astype(np.uint8)).long(), \
               torch.from_numpy(label.astype(np.uint8)).long()


def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)
    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])


def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')


class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x


class RandomFlip(object):
    """Flip randomly the image.
    """

    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()


class GaussianNoise(object):
    """Add gaussian noise to the image.
    """

    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x


from PIL import Image, ImageOps, ImageFilter


class ToTensor(object):
    """Transform the image to tensor.
    """

    def __call__(self, x):
        x = torch.from_numpy(x)
        return x


class RandomRotate(object):
    def __call__(self, x):
        c, h, w = x.shape

if __name__ == '__main__':
    import warnings
    import matplotlib.pyplot as plt
    warnings.filterwarnings('ignore')
    # dataset_loader = DataLoader(VAL_CLS_by_seg_v2(r"../data/miccai2023/version4_fold1/train.txt", "./EST",2), batch_size=1)
    dataset = CLS_base_MOCO_train_seg_v2(r"/data/data_disk/zhaobt/project/tumor_cls/data/miccai2023/version4_fold1/train.txt",)
    # my_sample = moco_sampler(dataset, batch_size=32, iters=100)
    dataset_loader = DataLoader(dataset, num_workers=32, shuffle=True, batch_size=32)
    res = 0
    for i,data in enumerate(dataset_loader):
        print("k size", i, data[-3])

        # if  data[-3]==1:
        #     res += 1
        # print(res)
        pass
        # img,  label, _ = data
        # print(label)
        # plt.subplot(241)
        # plt.imshow(img[0,0,10,...].numpy())
        # plt.subplot(242)
        # # plt.imshow(mask[0,10 ,...].numpy())
        # plt.subplot(243)
        # plt.imshow(img[1,0,50,...].numpy())
        # plt.subplot(244)
        # # plt.imshow(mask[1,50,...].numpy())
        # plt.subplot(245)
        # plt.imshow(img[2,0,40,...].numpy())
        # plt.subplot(246)
        # # plt.imshow(mask[2,40,...].numpy())
        # plt.subplot(247)
        # plt.imshow(img[3,0,40,...].numpy())
        # plt.subplot(248)
        # # plt.imshow(mask[3,40,...].numpy())
        # plt.show()
        # break
        # import pdb
        # pdb.set_trace()
        # if i > 10000:
        #     break

# if __name__ == '__main__':
#     import pdb, tqdm
#     train_cls_dataset = CLS_base_multi_task_v2(r"./data/version4/fold1/train.txt", 1, 2)
#     for i in tqdm.tqdm(range(1000), ncols=20):
#         data = next(train_cls_dataset)
#         print(torch.sum(data[1])/(64*112*96))
#         # pdb.set_trace()
#         if data[0].size()[2] != 64 or data[0].size()[3] != 112 or data[0].size()[4] != 96 \
#                 or data[1].size()[1] != 64 or data[1].size()[2] != 112 or data[1].size()[3] != 96:
#             pdb.set_trace()
#
#     # val_cls_dataset = DataLoader(data_generator.Cls_base(cfg.data.val_txt), batch_size=1, shuffle=False)
#     # data_dir_un = r"D:\dataset\tumor_classification\preprocessed_dataset1\dataset1_unlabeled.h5"
#     # unlabeled_f = h5py.File(data_dir_un, 'r')
#     # unlabeled_dataset = unlabeled_f['img']
#     #
#     # data_dir_labeled_train = r"D:\dataset\tumor_classification\preprocessed_dataset1\dataset1_labeled_train.h5"
#     # labeled_f = h5py.File(data_dir_labeled_train, 'r')
#     # train_labeled_img_dataset = labeled_f['img']
#     # train_labeled_mask_dataset = labeled_f['mask']
#     #
#     # transform_train = transforms.Compose([
#     #     RandomPadandCrop(32),
#     #     RandomFlip(),
#     #     ToTensor(),
#     # ])
#     # a = labeled_f['img'][[1, 2, 3, ], ...]
#     # dataset = generator_2D_labeled(labeled_f['img'], labeled_f['mask'], batch_size=10)
#     #
#     # x, mask, label = next(dataset)
#     # print(x.size())
#     # print(mask.size())
#     # print(label.size())
#     #
#     # val_dataset = generator_2D_val(labeled_f['img'], labeled_f['mask'])
#     # val_dataset = DataLoader(val_dataset, batch_size=2)
#     # for i, data in enumerate(val_dataset):
#     #     img, mask, _ = data
#     #     print(img.size())
#     #     print(mask.size())
#     #     print(_.size())
#     #     break

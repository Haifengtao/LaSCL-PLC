#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   file_io.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/27 13:48   Bot Zhao      1.0         None
"""

# import lib
import SimpleITK as sitk
import os
import numpy as np
import nibabel
import sys
import glob
import json, shutil, random
import matplotlib.pyplot as plt
sys.path.append("/home1/zhaobt/PycharmProjects/tumor_cls/")
from utils import misc
from nets import Losses
from utils import data_generator
from nets import unet
from utils import img_utils
from utils import logger
from utils import model_io


def save_nii_array(array, save_dir, temp=None):
    """
    save a 3d array ti nii file.
    :param array:
    :param save_dir:
    :param temp:
    :return:
    """
    # array = array.transpose((2, 1, 0))
    image = sitk.GetImageFromArray(array)
    if temp is not None:
        image.SetSpacing(temp.GetSpacing())
        image.SetOrigin(temp.GetOrigin())
        image.SetDirection(temp.GetDirection())
    sitk.WriteImage(image, save_dir)




# sitk.sitkU
def read_array_nii(input_dir):
    image = sitk.ReadImage(input_dir)
    array = sitk.GetArrayFromImage(image)
    # array = array.transpose((2, 1, 0))
    return image, array


def draw_fig(pred, tumor):
    pred = pred.squeeze()
    tumor = tumor.squeeze()
    indexs = np.nonzero(tumor)
    minx, miny, minz = np.min(indexs[0]), np.min(indexs[1]), np.min(indexs[2])
    maxx, maxy, maxz = np.max(indexs[0]), np.max(indexs[1]), np.max(indexs[2])
    fig = plt.figure()
    plt.subplot(231)
    plt.imshow(pred[(minx+maxx)//2, :, :])
    plt.colorbar()
    plt.subplot(232)
    plt.imshow(pred[:, :, (minz + maxz) // 2])
    plt.colorbar()
    plt.subplot(233)
    plt.imshow(pred[:, :, (minz + maxz) // 2])
    plt.colorbar()
    plt.subplot(234)
    plt.imshow(tumor[(minx+maxx)//2, :, :])
    plt.colorbar()
    plt.subplot(235)
    plt.imshow(tumor[:, :, (minz + maxz) // 2])
    plt.colorbar()
    plt.subplot(236)
    plt.imshow(tumor[:, :, (minz + maxz) // 2])
    plt.colorbar()
    # plt.savefig(path)
    return fig


def draw_fig_2d(x, pred, tumor):
    x = x.squeeze()
    pred = pred.squeeze()
    tumor = tumor.squeeze()
    fig = plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.imshow(pred)
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(tumor)
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(x, cmap="gray")
    plt.colorbar()
    # plt.savefig(path)
    return fig


def dicom2nii(rootdir, cur_name, new_name):
    if not os.listdir(rootdir):
        return
    if not os.path.isdir(os.path.join(rootdir, os.listdir(rootdir)[0])):
        outdir = rootdir.replace(cur_name, new_name)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        os.system("D: && cd D:\jupyter\cest_pipeline\external tools && dcm2niix -f " +
                  "%f_%p_%t_%s" + ' -i n -l y -p y -x n -v 0 -z y -o ' + outdir
                  + " " + rootdir)
    else:
        for i in os.listdir(rootdir):
            print(i)
            dicom2nii(os.path.join(rootdir, i), cur_name, new_name)


def load_module_from_disk(file):
    from importlib.machinery import SourceFileLoader
    func = SourceFileLoader('module.name', file).load_module()
    return func


def save_nii_fromH5(img_arr, meta_info):
    import pdb
    ref_path = meta_info["img_path"]
    ref_path = ref_path.replace("\\\\", "/")
    true_ref_path = ref_path.replace("D:/dataset/tumor_classification","/share/inspurStorage/home1/zhaobt/data/tumor_classification/")
    mask_path = true_ref_path.replace(".nii.gz", "_pred_mask_with_tumor.nii.gz")
    # print(ref_path)
    # print(true_ref_path)
    img = sitk.ReadImage(true_ref_path)
    # pad and resize
    x, y, z = img.GetSize()
    if x <= 512 and y <= 512:
        minx, miny = (512-x)//2, (512-y)//2
        maxx, maxy = ((512-x)//2)+x, ((512-y)//2)+y
        # pdb.set_trace()
        img_arr = img_arr[:, miny:maxy, minx:maxx]
        save_nii_array(img_arr, mask_path, img)
    else:
        img_arr = sitk.GetImageFromArray(img_arr)
        temp_mask = img_utils.resize_image_size(img_arr, (x, y, z))
        img_arr = sitk.GetArrayFromImage(temp_mask)
        save_nii_array(img_arr, mask_path, img)
    print(mask_path)
    root_dir, name = os.path.split(true_ref_path)
    # return mask_path, os.path.join(root_dir, "mask_"+name)


def get_roi_dataset(img_dir, mask_dir):
    try:
        image, img_arr = read_array_nii(img_dir)
        mask, mask_arr = read_array_nii(mask_dir)
        mask_arr = img_utils.rm_small_cc(mask_arr.astype("int32"), rate=0.5)
        st, ed = img_utils.get_bbox(mask_arr)
        img_roi = img_utils.crop_img(img_arr, st, ed)
        save_nii_array(img_roi, img_dir.replace('.nii.gz', "_roi.nii.gz"), image)
        return img_roi.shape, image.GetSize()
    except:
        return [0,0,0], [0,0,0]


def get_file_list(data_dir):
    import pandas as pd
    data = pd.read_csv(data_dir)
    pathes = []
    infos = []
    for i in data.iterrows():
        if i[1]["T1CE_num"] >= 1:
            ref_path = i[1]["T1CE"][2:-2]
            ref_path = ref_path.replace("\\\\", "/")
            true_ref_path = ref_path.replace("D:/dataset/tumor_classification",
                                             "/share/inspurStorage/home1/zhaobt/data/tumor_classification/")
            if not os.path.isfile(true_ref_path):
                print("There is a file bug!!!")
            pathes.append(true_ref_path)
            # import pdb
            # pdb.set_trace()
            info = json.loads(i[1]["T1CE_info"])
            infos.append(info)
        # if i[1]["T1CE_num"] == 1 and i[1]["labeled_T1CE"] != "0":
        #     # ref_path = i[1]["T1CE"][2:-2]
        #     ref_path = i[1]["labeled_T1CE"]
        #     ref_path = ref_path.replace("\\\\", "/")
        #     true_ref_path = ref_path.replace("D:/dataset/tumor_classification",
        #                                      "/share/inspurStorage/home1/zhaobt/data/tumor_classification/")
        #     if not os.path.isfile(true_ref_path):
        #         print("There is a file bug!!!")
        #     pathes.append(true_ref_path)
    # import pdb
    # pdb.set_trace()
    return pathes, infos


def save_img_from_3d(st, et, img, save_dir):
    p1 = [int((i + j)/2) for i, j in zip(st, et)]
    plt.subplot(221)
    plt.imshow(img[p1[0], :, :], cmap="gray")
    plt.subplot(222)
    plt.imshow(img[:, p1[1], :])
    plt.subplot(223)
    plt.imshow(img[:, :, p1[2]])
    plt.savefig(save_dir)


def collect_mask():
    data = {}
    data_dir = "/share/inspurStorage/home1/zhaobt/data/tumor_classification/tumor_dataset2"
    data1 = glob.glob(os.path.join(data_dir, "*/*/*/mask*.nii.gz"))
    print(len(data1))
    # print(len(data2))
    data_txt = "/share/inspurStorage/home1/zhaobt/data/tumor_classification/preprocessed_2/data_with_tumor.txt"
    for i in data1:
        dir, file_name = os.path.split(i)
        if file_name[:5] != "mask_":
            print(i)
    # with open(data_txt, "w") as f:
    # for i in data1:
    #     dir, file_name = os.path.split(i)
    #     outdir = "/".join(dir.split("/")[:-1])
    #     shutil.copyfile(i, os.path.join(outdir, file_name.replace("mask", "tumor")))
    #     print(file_name)
    #     if os.path.isfile(os.path.join(outdir, file_name[5:])):
    #         img_path = os.path.join(outdir, file_name[5:])
    #         print(img_path)
    #
    #     for j in os.listdir(dir):
    #         os.remove(os.path.join(dir, j))
    #     os.removedirs(dir)
        # break
    data = glob.glob(os.path.join(data_dir, "*/*/tumor*.nii.gz"))
    with open(data_txt, "w") as f:
        for i in data:
            dir, file_name = os.path.split(i)
            print(os.path.join(dir, file_name[6:]).replace(".nii.gz", "_roi_resize_crop.nii.gz"))
            if os.path.isfile(os.path.join(dir, file_name[6:]).replace(".nii.gz", "_roi_resize_crop.nii.gz")):
                img_path = os.path.join(dir, file_name[6:]).replace(".nii.gz", "_roi_resize_crop.nii.gz")
                if "hemangioblastoma" in i:  # 血管细胞瘤
                    f.write(img_path + "|0" + "\n")
                elif "angiocavernoma" in i:  # 海绵状血管瘤
                    f.write(img_path + "|1" + "\n")
                elif "III-IV_glioma" in i:  # 3-4型胶质瘤
                    f.write(img_path + "|2" + "\n")
                elif "pilocytic_astrocytoma" in i:  # 星型细胞瘤
                    f.write(img_path + "|3" + "\n")
            else:
                print(img_path)


def split_data():
    data_dir = "../data/version2/dataset_merged.txt"
    data = {"hemangioblastoma": 0, "angiocavernoma": 0, "III-IV_glioma": 0, "pilocytic_astrocytoma": 0}
    with open(data_dir, "r") as f:
        strs = f.readlines()
        random.shuffle(strs)

        train_txt = open("../data/version2/train_merged.txt", "w")
        val_txt = open("../data/version2/val_merged.txt", "w")
        test_txt = open("../data/version2/test_merged.txt", "w")
        for idx, i in enumerate(strs):
            img_path = i.split("|")[0]
            if "hemangioblastoma" in i:  # 血管细胞瘤
                data["hemangioblastoma"] += 1
                print(strs[idx])
                img_path += "|0"
            elif "angiocavernoma" in i:  # 海绵状血管瘤
                data["angiocavernoma"] += 1
                img_path += "|1"
            elif "III-IV_glioma" in i:  # 3-4型胶质瘤
                data["III-IV_glioma"] += 1
                img_path += "|2"
            elif "pilocytic_astrocytoma" in i:  # 星型细胞瘤
                data["pilocytic_astrocytoma"] += 1
                img_path += "|3"
            if idx < 200:
                train_txt.write(img_path+"\n")
            elif idx < 220:
                val_txt.write(img_path+"\n")
            else:
                test_txt.write(img_path+"\n")
        train_txt.close()
        val_txt.close()
        test_txt.close()
    json.dump(data, open("../data/version2/stat.json", "w"))


def split_data_2():
    data_dir = "../data/version2/dataset_merged.txt"
    data = {"hemangioblastoma": 0, "angiocavernoma": 0, "III-IV_glioma": 0, "pilocytic_astrocytoma": 0}
    with open(data_dir, "r") as f:
        strs = f.readlines()
        random.shuffle(strs)

        train_txt = open("../data/version2/train_merged_2.txt", "w")
        val_txt = open("../data/version2/val_merged_2.txt", "w")
        test_txt = open("../data/version2/test_merged_2.txt", "w")
        for idx, i in enumerate(strs):
            img_path = i.split("|")[0]
            if "hemangioblastoma" in i:  # 血管细胞瘤
                data["hemangioblastoma"] += 1
                print(strs[idx])
                img_path += "|1"
            elif "angiocavernoma" in i:  # 海绵状血管瘤
                data["angiocavernoma"] += 1
                img_path += "|0"
            elif "III-IV_glioma" in i:  # 3-4型胶质瘤
                data["III-IV_glioma"] += 1
                img_path += "|0"
            elif "pilocytic_astrocytoma" in i:  # 星型细胞瘤
                data["pilocytic_astrocytoma"] += 1
                img_path += "|0"
            if idx < 200:
                train_txt.write(img_path + "\n")
            elif idx < 220:
                val_txt.write(img_path + "\n")
            else:
                test_txt.write(img_path + "\n")
        train_txt.close()
        val_txt.close()
        test_txt.close()
    json.dump(data, open("../data/version2/stat.json", "w"))


def split_data_3():
    data_dir = "../data/version4/boceng_dataset.txt"
    data = {"hemangioblastoma": 0, "angiocavernoma": 0, "III-IV_glioma": 0, "pilocytic_astrocytoma": 0}
    with open(data_dir, "r") as f:
        strs = f.readlines()
        random.shuffle(strs)
        num = len(strs)//5
        test_1 = strs[:num]
        train_1 = strs[num:]

        test_2 = strs[num:num*2]
        train_2 = strs[:num] + strs[num*2:]

        test_3 = strs[num*2:num * 3]
        train_3 = strs[:num*2] + strs[num * 3:]

        test_4 = strs[num*3:num * 4]
        train_4 = strs[:num*3] + strs[num * 4:]

        test_5 = strs[num * 4:]
        train_5 = strs[:num*4] + strs[num * 5:]

        tests = [test_1, test_2, test_3, test_4, test_5]
        trains = [train_1, train_2, train_3, train_4, train_5]
        idx = 1
        for train, test in zip(trains, tests):
            print(str(idx) + "train:" + str(len(train)) + " test:" + str(len(test)))
            train_file = open("../data/version4/fold" + str(idx) + "/train.txt", "w")
            test_file = open("../data/version4/fold" + str(idx) + "/test.txt", "w")
            idx += 1
            for i in train:
                train_file.write(i)
            for i in test:
                test_file.write(i)
            train_file.close()
            test_file.close()

        for idx, i in enumerate(strs):
            img_path = i.split("|")[0]
            if "hemangioblastoma" in i:  # 血管细胞瘤
                data["hemangioblastoma"] += 1
                img_path += "|1"
            elif "angiocavernoma" in i:  # 海绵状血管瘤
                data["angiocavernoma"] += 1
                img_path += "|0"
            elif "III-IV_glioma" in i:  # 3-4型胶质瘤
                data["III-IV_glioma"] += 1
                img_path += "|0"
            elif "pilocytic_astrocytoma" in i:  # 星型细胞瘤
                data["pilocytic_astrocytoma"] += 1
                img_path += "|0"
    json.dump(data, open("../data/version4/boceng_stat.json", "w"))

# split_data_3()

# split_data_2()

# collect_mask()

# if __name__ == '__main__':
#     # root_die = r"D:\dataset\tumors_renamed_data\classified_data\general_hospital\pilocytic_astrocytoma"
#     # dicom2nii(root_die, "tumors_renamed_data", "tumor_classification")
#     import tqdm
#
#     data_dir = r"/share/inspurStorage/home1/zhaobt/data/tumor_classification/preprocessed_1/dataset1_summary_v1.csv"
#     pathes, infos = get_file_list(data_dir)
#     title = ["file_name", "size_x", "size_y", "size_z", "spacing_x", "spacing_y", "spacing_z"]
#     # title = ["file_name", "spacing_x", "spacing_y", "spacing_z"]
#     log = logger.Logger_csv(title, data_dir.replace('dataset1_summary_v1.csv', ''), "T1CE_spacing_info_temp.csv")
#     print("running")
#     # print(pathes)
#     # img_dir = r"D:\dataset\tumor_classification\temp\T1+_T1_MPRAGE_TRA_iso1.0_20130923131342_2.nii.gz"
#     # mask_dir = img_dir.replace(".nii.gz", "_pred_mask_baseling.nii.gz")
#     for idx, img_dir in tqdm.tqdm(enumerate(pathes[145:146]), ncols=10):
#         # spacing = infos[idx]['img_spacing']
#         # log.update({"file_name": img_dir,"spacing_x": spacing[0], "spacing_y": spacing[1], "spacing_z": spacing[2]})
#         #
#         import pdb
#         pdb.set_trace()
#
#         print(img_dir)
#         mask_dir = img_dir.replace(".nii.gz", "_pred_mask_baseling.nii.gz")
#         try:
#             [z, x, y], [sx, sy, sz] = get_roi_dataset(img_dir, mask_dir)
#             log.update({"file_name": img_dir, "size_x": x, "size_y": y, "size_z": z,
#                         "spacing_x": sx, "spacing_y": sy, "spacing_z": sz})
#         except:
#             log.update({"file_name": img_dir, "size_x": 0, "size_y": 0, "size_z": 0})
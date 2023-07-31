#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   preprocess.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, 2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/31 23:28   Bot Zhao      1.0         None
"""

# import lib
import SimpleITK as sitk
import os
import numpy as np
import sys
import tqdm
import pdb
import random
import pandas as pd

sys.path.append("/home1/zhaobt/PycharmProjects/tumor_cls/")
from utils import img_utils
from utils import logger
from utils import file_io


def preprocess(img_dir, mask_dir):
    """

    :param img_dir:
    :param mask_dir:
    :return:
    """
    # 0. read image
    # if os.path.exists(img_dir[:-3]):
    #     # try:
    #     #     os.rmdir(img_dir[:-3])
    #     # except:
    #     #     os.remove(img_dir[:-3])
    #     os.system("rm -r " + img_dir[:-3])
    if not os.path.isfile(img_dir) or not os.path.isfile(mask_dir):
        return img_dir.replace('.nii.gz', '_roi_resize_crop.nii.gz'), -1, [0, 0, 0], [0, 0, 0]
    try:
        img, img_arr = file_io.read_array_nii(img_dir)
        # 1. crop image
        mask, mask_arr = file_io.read_array_nii(mask_dir)
        mask_arr = img_utils.rm_small_cc(mask_arr.astype("int32"), rate=0.3)
        mask_arr[mask_arr != 0] = 1
        st, ed = img_utils.get_bbox(mask_arr)
        img_roi = img_utils.crop_img(img_arr, st, ed)
        roi_shape = img_roi.shape
        spacing = mask.GetSpacing()
        file_io.save_nii_array(img_roi, img_dir.replace('.nii.gz', "_roi.nii.gz"), img)

        # 2. resample to the same spacing
        img_roi = sitk.GetImageFromArray(img_roi)
        img_roi.SetSpacing(img.GetSpacing())
        itkimg_Resampled = img_utils.resize_image_itk(img_roi, (1, 1, 1), resamplemethod=sitk.sitkLinear)
        sitk.WriteImage(itkimg_Resampled, img_dir.replace('.nii.gz', "_resampled_roi.nii.gz"))

        # 3. crop or padding to the same size
        # no crop: Because the background is not the value zero.
        img_Resampled = sitk.GetArrayFromImage(itkimg_Resampled)
        z, y, x = img_Resampled.shape
        # pad and crop
        max_x, max_y, max_z = 96, 112, 64

        if y <= max_y:
            cropped_img = np.pad(img_Resampled, ((0, 0), ((max_y - y) // 2, (max_y - y) - (max_y - y) // 2),
                                                 (0, 0),), "constant", constant_values=((0, 0), (0, 0), (0, 0)))
        else:
            st_y = (y - max_y) // 2
            cropped_img = img_Resampled[:, st_y:(st_y + max_y), :]

        if x <= max_x:
            cropped_img = np.pad(cropped_img, ((0, 0), (0, 0),
                                               ((max_x - x) // 2, (max_x - x) - (max_x - x) // 2)), "constant",
                                 constant_values=((0, 0), (0, 0), (0, 0)))
        else:
            st_x = (x - max_x) // 2
            cropped_img = cropped_img[..., st_x:(st_x + max_x)]

        if z <= max_z:
            cropped_img = np.pad(cropped_img, (((max_z - z) // 2, (max_z - z) - (max_z - z) // 2), (0, 0), (0, 0),),
                                 "constant", constant_values=((0, 0), (0, 0), (0, 0)))
        else:
            st_z = (z - max_z) // 2
            cropped_img = cropped_img[st_z:(st_z + max_z), ...]
        file_io.save_nii_array(cropped_img, img_dir.replace('.nii.gz', '_roi_resize_crop.nii.gz'), itkimg_Resampled)

        if "hemangioblastoma" in img_dir:
            type = 1
        else:
            type = 0
        # print(roi_shape)
        return img_dir.replace('.nii.gz', '_roi_resize_crop.nii.gz'), type, roi_shape, spacing

    except:
        return img_dir.replace('.nii.gz', '_roi_resize_crop.nii.gz'), -1, [0, 0, 0], [0, 0, 0]


def preprocess_with_tumor(img_dir, img2_dir, mask_dir):
    """

    :param img_dir:
    :param mask_dir:
    :return:
    """
    # 0. read image
    # if os.path.exists(img_dir[:-3]):
    #     # try:
    #     #     os.rmdir(img_dir[:-3])
    #     # except:
    #     #     os.remove(img_dir[:-3])
    #     os.system("rm -r " + img_dir[:-3])
    if not os.path.isfile(img_dir) or not os.path.isfile(mask_dir):
        print("not found {} \n {}".format(img_dir, mask_dir))
        return img_dir.replace('.nii.gz', '_roi_resize_crop.nii.gz'), -1, [0, 0, 0], [0, 0, 0]
    try:
        img, img_arr = file_io.read_array_nii(img_dir)
        img2, img2_arr = file_io.read_array_nii(img2_dir)

        # 1. crop image
        mask, mask_arr = file_io.read_array_nii(mask_dir)
        mask_arr[mask_arr != 0] = 1
        mask_arr = img_utils.rm_small_cc(mask_arr.astype("int32"), rate=0.3)
        st, ed = img_utils.get_bbox(mask_arr)
        img_roi = img_utils.crop_img(img_arr, st, ed)
        img2_roi = img_utils.crop_img(img2_arr, st, ed)
        roi_shape = img_roi.shape
        spacing = mask.GetSpacing()
        # file_io.save_img_from_3d(st, ed, img_arr, os.path.join(test_path, temp_id))
        file_io.save_nii_array(img_roi, img_dir.replace('.nii.gz', "_roi.nii.gz"), img)

        # 2. resample to the same spacing
        img_roi = sitk.GetImageFromArray(img_roi)
        img2_roi = sitk.GetImageFromArray(img2_roi)
        img2_roi.SetSpacing(img2.GetSpacing())
        img_roi.SetSpacing(img.GetSpacing())
        itkimg_Resampled = img_utils.resize_image_itk(img_roi, (1, 1, 1), resamplemethod=sitk.sitkLinear)
        itkimg2_Resampled = img_utils.resize_image_itk(img2_roi, (1, 1, 1), resamplemethod=sitk.sitkLinear)
        sitk.WriteImage(itkimg_Resampled, img_dir.replace('.nii.gz', "_resampled_roi.nii.gz"))
        # sitk.WriteImage(itkimg2_Resampled, img2_dir.replace('.nii.gz', "_resampled_roi.nii.gz"))

        # 3. crop or padding to the same size
        # no crop: Because the background is not the value zero.
        # pdb.set_trace()

        img_Resampled = sitk.GetArrayFromImage(itkimg_Resampled)
        img2_Resampled = sitk.GetArrayFromImage(itkimg2_Resampled)
        z, y, x = img_Resampled.shape
        # pad and crop
        max_x, max_y, max_z = 96, 112, 64

        if y <= max_y:
            cropped_img = np.pad(img_Resampled, ((0, 0), ((max_y - y) // 2, (max_y - y) - (max_y - y) // 2),
                                                 (0, 0),), "constant", constant_values=((0, 0), (0, 0), (0, 0)))
            cropped_img2 = np.pad(img2_Resampled, ((0, 0), ((max_y - y) // 2, (max_y - y) - (max_y - y) // 2),
                                                   (0, 0),), "constant", constant_values=((0, 0), (0, 0), (0, 0)))
        else:
            st_y = (y - max_y) // 2
            cropped_img = img_Resampled[:, st_y:(st_y + max_y), :]
            cropped_img2 = img2_Resampled[:, st_y:(st_y + max_y), :]

        if x <= max_x:
            cropped_img = np.pad(cropped_img, ((0, 0), (0, 0),
                                               ((max_x - x) // 2, (max_x - x) - (max_x - x) // 2)), "constant",
                                 constant_values=((0, 0), (0, 0), (0, 0)))
            cropped_img2 = np.pad(cropped_img2, ((0, 0), (0, 0),
                                                 ((max_x - x) // 2, (max_x - x) - (max_x - x) // 2)), "constant",
                                  constant_values=((0, 0), (0, 0), (0, 0)))
        else:
            st_x = (x - max_x) // 2
            cropped_img = cropped_img[..., st_x:(st_x + max_x)]
            cropped_img2 = cropped_img2[..., st_x:(st_x + max_x)]

        if z <= max_z:
            cropped_img = np.pad(cropped_img, (((max_z - z) // 2, (max_z - z) - (max_z - z) // 2), (0, 0), (0, 0),),
                                 "constant", constant_values=((0, 0), (0, 0), (0, 0)))
            cropped_img2 = np.pad(cropped_img2, (((max_z - z) // 2, (max_z - z) - (max_z - z) // 2), (0, 0), (0, 0),),
                                  "constant", constant_values=((0, 0), (0, 0), (0, 0)))
        else:
            st_z = (z - max_z) // 2
            cropped_img = cropped_img[st_z:(st_z + max_z), ...]
            cropped_img2 = cropped_img2[st_z:(st_z + max_z), ...]
        file_io.save_nii_array(cropped_img, img_dir.replace('.nii.gz', '_roi_resize_crop.nii.gz'), itkimg_Resampled)
        file_io.save_nii_array(cropped_img2, img2_dir.replace('.nii.gz', '_roi_resize_crop.nii.gz'), itkimg2_Resampled)

        if "hemangioblastoma" in img_dir:
            type = 1
        else:
            type = 0
        # print(roi_shape)
        return img_dir.replace('.nii.gz', '_roi_resize_crop.nii.gz'), type, roi_shape, spacing

    except Exception as e:
        print(e)
        return img_dir.replace('.nii.gz', '_roi_resize_crop.nii.gz'), -1, [0, 0, 0], [0, 0, 0]


def preprocess_tumor_old(img_dir, img2_dir, test_path):
    """
    只选取肿瘤
    :param img_dir:   raw nii
    :param mask_dir:  tumor mask
    :return:
    """

    # if not os.path.isfile(img_dir) or not os.path.isfile(img2_dir):
    #     print("not found")
    #     return img_dir.replace('.nii.gz', '_roi_resize_crop.nii.gz'), -1, [0, 0, 0], [0, 0, 0]
    try:
        img, img_arr = file_io.read_array_nii(img_dir)
        img2, img2_arr = file_io.read_array_nii(img2_dir)
        img2_arr[img2_arr != np.max(img2_arr)] = 0

        # 1. crop image
        st, ed = img_utils.get_bbox(img2_arr, pad=3)
        temp_id = img_dir.replace('/', "_")+".png"
        file_io.save_img_from_3d(st, ed, img_arr, os.path.join(test_path, temp_id))

        img_roi = img_utils.crop_img(img_arr, st, ed)
        img2_roi = img_utils.crop_img(img2_arr, st, ed)
        spacing = img2.GetSpacing()
        file_io.save_nii_array(img_roi, img_dir.replace('.nii.gz', "_tumor_roi.nii.gz"), img)

        # 2. resample to the same spacing
        img_roi = sitk.GetImageFromArray(img_roi)
        img2_roi = sitk.GetImageFromArray(img2_roi)
        img2_roi.SetSpacing(img2.GetSpacing())
        img_roi.SetSpacing(img.GetSpacing())
        itkimg_Resampled = img_utils.resize_image_itk(img_roi, (1, 1, 1), resamplemethod=sitk.sitkLinear)
        itkimg2_Resampled = img_utils.resize_image_itk(img2_roi, (1, 1, 1), resamplemethod=sitk.sitkLinear)
        sitk.WriteImage(itkimg_Resampled, img_dir.replace('.nii.gz', "_tumor_resampled_roi.nii.gz"))


        # 3. crop or padding to the same size
        # no crop: Because the background is not the value zero.
        # pdb.set_trace()
        img_Resampled = sitk.GetArrayFromImage(itkimg_Resampled)
        img2_Resampled = sitk.GetArrayFromImage(itkimg2_Resampled)
        z, y, x = img_Resampled.shape
        # pad and crop
        max_x, max_y, max_z = 64, 64, 64

        if y <= max_y:
            cropped_img = np.pad(img_Resampled, ((0, 0), ((max_y - y) // 2, (max_y - y) - (max_y - y) // 2),
                                                 (0, 0),), "constant", constant_values=((0, 0), (0, 0), (0, 0)))
            cropped_img2 = np.pad(img2_Resampled, ((0, 0), ((max_y - y) // 2, (max_y - y) - (max_y - y) // 2),
                                                   (0, 0),), "constant", constant_values=((0, 0), (0, 0), (0, 0)))
        else:
            st_y = (y - max_y) // 2
            cropped_img = img_Resampled[:, st_y:(st_y + max_y), :]
            cropped_img2 = img2_Resampled[:, st_y:(st_y + max_y), :]

        if x <= max_x:
            cropped_img = np.pad(cropped_img, ((0, 0), (0, 0),
                                               ((max_x - x) // 2, (max_x - x) - (max_x - x) // 2)), "constant",
                                 constant_values=((0, 0), (0, 0), (0, 0)))
            cropped_img2 = np.pad(cropped_img2, ((0, 0), (0, 0),
                                                 ((max_x - x) // 2, (max_x - x) - (max_x - x) // 2)), "constant",
                                  constant_values=((0, 0), (0, 0), (0, 0)))
        else:
            st_x = (x - max_x) // 2
            cropped_img = cropped_img[..., st_x:(st_x + max_x)]
            cropped_img2 = cropped_img2[..., st_x:(st_x + max_x)]

        if z <= max_z:
            cropped_img = np.pad(cropped_img, (((max_z - z) // 2, (max_z - z) - (max_z - z) // 2), (0, 0), (0, 0),),
                                 "constant", constant_values=((0, 0), (0, 0), (0, 0)))
            cropped_img2 = np.pad(cropped_img2, (((max_z - z) // 2, (max_z - z) - (max_z - z) // 2), (0, 0), (0, 0),),
                                  "constant", constant_values=((0, 0), (0, 0), (0, 0)))
        else:
            st_z = (z - max_z) // 2
            cropped_img = cropped_img[st_z:(st_z + max_z), ...]
            cropped_img2 = cropped_img2[st_z:(st_z + max_z), ...]
        file_io.save_nii_array(cropped_img, img_dir.replace('.nii.gz', '_tumor_roi_resize_crop.nii.gz'), itkimg_Resampled)
        file_io.save_nii_array(cropped_img2, img2_dir.replace('.nii.gz', '_tumor_roi_resize_crop.nii.gz'), itkimg2_Resampled)
        return [0,0,0], spacing

    except:
        print("error!")
        # return [0, 0, 0], [0, 0, 0]
        return img_dir.replace('.nii.gz', '_tumor_roi_resize_crop.nii.gz'), -1, [0, 0, 0], [0, 0, 0]


def preprocess_tumor(img_dir, img2_dir):
    """
    只选取肿瘤
    :param img_dir:   raw nii
    :param mask_dir:  tumor mask
    :return:
    """

    # if not os.path.isfile(img_dir) or not os.path.isfile(img2_dir):
    #     print("not found")
    #     return img_dir.replace('.nii.gz', '_roi_resize_crop.nii.gz'), -1, [0, 0, 0], [0, 0, 0]
    try:
        img, img_arr = file_io.read_array_nii(img_dir.replace('.nii.gz', '_roi_resize_crop.nii.gz'))
        img2, img2_arr = file_io.read_array_nii(img2_dir.replace('.nii.gz', '_roi_resize_crop.nii.gz'))
        img_arr[img2_arr == 0] = 0
        file_io.save_nii_array(img_arr, img_dir.replace('.nii.gz', '_tumor_roi_resize_crop.nii.gz'),
                               img)
        return [0, 0, 0], [0, 0, 0]

    except:
        return [0, 0, 0], [0, 0, 0]
        # return img_dir.replace('.nii.gz', '_tumor_roi_resize_crop.nii.gz'), -1, [0, 0, 0], [0, 0, 0]


# def main():
#     # test_path = "/share/inspurStorage/home1/zhaobt/data/tumor_classification/preprocessed_1/check_pic"
#     test_path = "/share/inspurStorage/home1/zhaobt/data/tumor_classification/preprocessed_2/check_pic_2"
#     with open("/share/inspurStorage/home1/zhaobt/data/tumor_classification/preprocessed_2/data_with_tumor.txt", "r") as f:
#         pathes = f.readlines()
#         print(len(pathes))
#         res = []
#         roi_shapes = dict()
#         for i in tqdm.tqdm(pathes):
#             case = i.strip().split("|")[0]
#             if os.path.isfile(case):
#                 res.append(i.strip())
#                 root_dir, file_name = os.path.split(case.replace("_roi_resize_crop.nii.gz", ".nii.gz"))
#                 # print(root_dir)
#                 # if "cao_yi_wen" not in root_dir:
#                 #     continue
#                 print(root_dir)
#                 print(file_name)
#                 roi_shape, spacing = preprocess_tumor_old(case.replace("_roi_resize_crop.nii.gz", ".nii.gz"),
#                                                           os.path.join(root_dir, "tumor_"+file_name), test_path)
#                 roi_shapes[i] = list(roi_shape) + list(spacing)
#                 # log.update({"file_name": img_path, "type": type, "x": shape1[0], "y": shape1[1],
#                 #             'z': shape1[2], "x1": spacing[0], "y1": spacing[1], 'z1': spacing[2]})
#             else:
#                 print("not found!")
#         # pd.DataFrame(roi_shapes).to_csv("/share/inspurStorage/home1/zhaobt/data/tumor_classification/preprocessed_1"
#         #                                 "/tumor_shape_1.csv")
#         # import pdb
#         # pdb.set_trace()
# main()


def save_train_cls(data_dir):
    pathes, infos = file_io.get_file_list(data_dir)
    random.shuffle(pathes)
    # out_dir
    pos_num = 0
    neg_num = 0
    with open("../data/temp.txt", "w") as f:
        for i in tqdm.tqdm(pathes):
            if not os.path.isfile(i.replace('.nii.gz', '_roi_resize_crop.nii.gz')):
                continue
            _, temp = file_io.read_array_nii(i.replace('.nii.gz', '_roi_resize_crop.nii.gz'))
            if len(temp.shape) > 3:
                print(os.path.isfile(i.replace('.nii.gz', '_roi_resize_crop.nii.gz')))
            if "hemangioblastoma" in i:
                type = 1
                pos_num += 1
            else:
                type = 0
                neg_num += 1
            info = i.replace('.nii.gz', '_roi_resize_crop.nii.gz') + "|" + str(type)
            f.write(info + "\n")
        print(neg_num, pos_num)


def save_train_all():
    pass


if __name__ == '__main__':
    import tqdm
    from utils import file_io
    import pdb
    import random

    # img_dir = r"D:\dataset\tumor_classification\temp\T1+_T1_MPRAGE_TRA_iso1.0_20160108122907_9.nii.gz"
    # preprocess(img_dir, img_dir.replace('.nii.gz', '_pred_mask_baseling.nii.gz'))
    data_dir = r"/share/inspurStorage/home1/zhaobt/data/tumor_classification/preprocessed_2"
    # save_train_cls(data_dir)

    title = ["file_name", "type", "x", "y", 'z', "x1", "y1", 'z1', ]
    log = logger.Logger_csv(title, data_dir, "dataset2_cereb.csv")
    print("running")

    with open(os.path.join(data_dir, "dataset2_info_v2.txt"), "r") as f:
        pathes = f.readlines()
        print(len(pathes))
        res = []

        for i in tqdm.tqdm(pathes):
            case = i.strip().split(" ")[0]
            mask = i.strip().split(" ")[1]
            # if "/share/inspurStorage/home1/zhaobt/data/tumor_classification/tumor_dataset2/pilocytic_astrocytoma/lv_yan" not in case:
            #     continue
            if os.path.isfile(case):
                res.append(i.strip())
                # root_dir, file_name = os.path.split(case.replace("_roi_resize_crop.nii.gz", ".nii.gz"))
                root_dir, file_name = os.path.split(case)
                # print(root_dir)
                if "xue_jia_ming" not in root_dir:
                    continue
                print(case)
                # img_path, type, shape1, spacing \
                #     = preprocess_with_tumor(case.replace("_roi_resize_crop.nii.gz", ".nii.gz"),
                #                             os.path.join(root_dir, "tumor_"+file_name),
                #                             case.replace('_roi_resize_crop.nii.gz', '_pred_mask_baseline.nii.gz'))
                img_path, type, shape1, spacing \
                        = preprocess_with_tumor(case,
                                                os.path.join(root_dir, mask),
                                                case.replace('.nii.gz', '_pred_mask_baseline.nii.gz'))
                print({"file_name": img_path, "type": type, "x": shape1[0], "y": shape1[1],
                            'z': shape1[2], "x1": spacing[0], "y1": spacing[1], 'z1': spacing[2]})
                log.update({"file_name": img_path, "type": type, "x": shape1[0], "y": shape1[1],
                            'z': shape1[2], "x1": spacing[0], "y1": spacing[1], 'z1': spacing[2]})
            else:
                print("not found!")

    # print(len(res))
    # pathes = [r'D:\dataset\tumor_classification\temp\temp\T1+_T1_MPRAGE_TRA_iso1.0_20191204111919_3.nii.gz']
    # for idx, img_dir in tqdm.tqdm(enumerate(pathes), ncols=10):
    #     print(img_dir)
    #     img_path, type = preprocess_with_tumor(img_dir, img_dir.replace('.nii.gz', '_pred_mask_baseling.nii.gz'))
    #     log.update({"img_name": img_path, "type": type})

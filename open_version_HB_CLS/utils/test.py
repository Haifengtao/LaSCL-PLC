#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright 2019-2021, ISTBI, Fudan University

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/21 13:45   Botao Zhao      1.0         None
'''

# import lib
import os, glob, json
import pandas as pd
import shutil
import glob
import h5py
import tqdm, random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def freeSurfer_bet(path, out_dir):
    """
    path: .nii.gz 文件路径，文档中均为.nii文件的路径
    """
    if not os.path.isfile(path):
        raise Exception("There is not a file: {}").format(path)

    cmd0 = 'source /home/zhang_istbi/.bashrc'
    cmd1 = "export SUBJECTS_DIR=" + out_dir + ";"
    try:
        #         将文件路径和文件名分离
        input_path = path
        data_dir, file_name = os.path.split(input_path)
        case_uid = file_name.replace('.nii.gz', '')
        # freesurfer环境配置、颅骨去除、未仿射对齐mpz转nii、仿射对齐、仿射对齐mpz转nii.gz格式
        # recon-all是颅骨去除的命令
        # mri_convert是进行格式转换，从mgz转到nii.gz，只是为了方便查看
        # --apply_transform：仿射对齐操作
        # 转格式

        save_path = os.path.join(out_dir, case_uid)
        # print("file name: ", save_path)
        cmd = cmd1 \
              + "recon-all -parallel -i " + input_path + " -autorecon1 -subjid " + case_uid + "&&" \
              + "mri_convert " + save_path + "/mri/brainmask.mgz " + save_path + "/mri/" + case_uid + "_bet.nii.gz;" \
              + "mri_convert " + save_path + "/mri/brainmask.mgz --apply_transform " + save_path + \
              "/mri/transforms/talairach.xfm -o " + save_path + "/mri/brainmask_affine.mgz&&" \
              + "mri_convert " + save_path + "/mri/brainmask_affine.mgz " + save_path + "/mri/" + case_uid + "_affine.nii.gz;"
        print(cmd)
        os.system(cmd)
    except Exception:
        print(Exception)
    return
pathes = ['../../../dataset/tumor_cls/tumor_dataset1/hemangioblastoma/luo_qin_fu/for_train/T1P.nii.gz',
          '../../../dataset/tumor_cls/tumor_dataset2/pilocytic_astrocytoma/he_jian_hui/for_train/T1P_biasC.nii.gz',
          '../../../dataset/tumor_cls/tumor_dataset2/hemangioblastoma/bian_xue_shan/for_train/T1P_biasC.nii.gz',
          '../../../dataset/tumor_cls/tumor_dataset2/hemangioblastoma/zhang_chi/for_train/T1P_biasC.nii.gz',
          '../../../dataset/tumor_cls/tumor_dataset2/III-IV_glioma/wang_ning/for_train/T1P_biasC.nii.gz']
for p in pathes[:1]:
    in_file = p
    data_dir, fileName = os.path.split(in_file)
    out_dir = data_dir.replace('../../../dataset/tumor_cls/tumor_dataset', '../../../dataset/tumor_cls/bet')
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    freeSurfer_bet(in_file, out_dir)
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   bet_field_bias.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright 2019-2021, ISTBI, Fudan University

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/3/30 23:30   Botao Zhao      1.0         None
'''

# import lib
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   bet_batches.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/4/10 23:29   Bot Zhao      1.0         None
"""

# import lib
import os, shutil
from tqdm import tqdm
import SimpleITK as sitk
import warnings
from nipype.interfaces.ants import N4BiasFieldCorrection


def fsl_bet(root_dir, out_dir):
    if not os.path.isdir(root_dir):
        raise Exception("There is not a path: {}").format(root_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    cases = os.listdir(root_dir)
    for case in tqdm(cases):
        input_path = os.path.join(root_dir, case)
        out_path = os.path.join(out_dir, case).replace('.nii.gz', 'bet.nii.gz')
        command = 'bet '+input_path+' '+out_path
        os.system(command)


def freeSurfer_bet(root_dir, out_dir, part):
    """
    root_dir: 目录下均为需要处理的.nii.gz 文件
    """
    if not os.path.isdir(root_dir):
        raise Exception("There is not a path: {}").format(root_dir)

    cases = os.listdir(root_dir)
    # 下面为freesurfer的环境配置命令
    # 数据所在的目录
    cmd1 = "export SUBJECTS_DIR=" + out_dir + ";"

    for case in cases[int(part)*300:(int(part)+1)*300]:
        if os.path.isfile(os.path.join(out_dir, case.replace('.nii.gz', '') + "/mri/"+case.replace('.nii.gz', '') + "_bet.nii")):
            continue
        try:
            # 将文件路径和文件名分离
            input_path = os.path.join(root_dir, case)
            # filename = os.path.splitext(filename)[0]  # 将文件名和扩展名分开，如果为.nii.gz，则认为扩展名是.gz
            case_uid = case.replace('.nii.gz', '')
            # freesurfer环境配置、颅骨去除、未仿射对齐mpz转nii、仿射对齐、仿射对齐mpz转nii.gz格式
            # recon-all是颅骨去除的命令
            # mri_convert是进行格式转换，从mgz转到nii.gz，只是为了方便查看
            # --apply_transform：仿射对齐操作
            # 转格式
            # filename = filename[:]  # 根据扩展名的不同，这里需要做更改，只保留文件名即可
            save_path = os.path.join(out_dir, case_uid)
            # print("file name: ", save_path)
            cmd = cmd1 \
                  + "recon-all -parallel -i " + input_path + " -autorecon1 -subjid " + case_uid + "&&" \
                  + "mri_convert " + save_path + "/mri/brainmask.mgz " + save_path + "/mri/" + case_uid + "_bet.nii.gz;" \
                  + "mri_convert " + save_path + "/mri/brainmask.mgz --apply_transform " + save_path + \
                  "/mri/transforms/talairach.xfm -o " + save_path + "/mri/brainmask_affine.mgz&&" \
                  + "mri_convert " + save_path + "/mri/brainmask_affine.mgz " + save_path + "/mri/" + case_uid + "_affine.nii.gz;"
            # print("cmd:\n",cmd)
            # print(save_path)
            os.system(cmd)

        except Exception:
            continue
        # break


def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection. If this fails, will then attempt to correct bias using SimpleITK
    :param in_file: nii文件的输入路径
    :param out_file: 校正后的文件保存路径名
    :return: 校正后的nii文件全路径名
    """
    # 使用N4BiasFieldCorrection校正MRI图像的偏置场
    if not os.path.isfile(in_file):
        raise Exception("No file")

    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
                                     "Will try using SimpleITK for bias field correction"
                                     " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
                                     " to your PATH system variable. (example: EXPORT PATH=${PATH}:/path/to/ants/bin)"))
        input_image = sitk.ReadImage(in_file, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_file)
        return os.path.abspath(out_file)


def normalize_image(in_file, out_file, bias_correction=True):
    # bias_correction：是否需要校正
    if bias_correction:
        correct_bias(in_file, out_file)
    else:
        shutil.copy(in_file, out_file)
    return out_file


def freeSurfer_bet_file(path, out_dir):
    """
    path: .nii.gz 文件路径，文档中均为.nii文件的路径
    """
    if not os.path.isfile(path):
        raise Exception("There is not a file: {}".format(path))

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

def collect_pathes(file):
    with open(file, 'r') as f:
        data = f.readlines()
    pathes = []
    for i in data:
        p = i.split(' ')[0].replace('_roi_resize_crop','').replace('/home/zhang_istbi/zhaobt/dataset/',
                                                                   '../../../dataset/')
        if os.path.isfile(p):
            pathes.append(p)
        else:
            print(p, 'error')
    return pathes


if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser("BET batches!")
    parse.add_argument('-m', "--mode", nargs="?", help="fsl or FreeSurfer(fs)")
    args = parse.parse_args()
    if args.mode == "fsl":
        fsl_bet(args.root_dir, args.out_dir)
    if args.mode == "fs":
        print("running FreeSurfer bet!")
        freeSurfer_bet(args.root_dir, args.out_dir, args.part)
    if args.mode == "fs_file":
        print("running FreeSurfer bet!")
        in_file = "../data/version_mix/fold1/test.txt"
        pathes = collect_pathes(in_file)
        print(pathes)
        for i in tqdm(pathes):
            out_file = i.replace('.nii.gz', '-biasC.nii.gz')
            normalize_image(i, out_file, bias_correction=True)

            in_file = i.replace('.nii.gz', '-biasC.nii.gz')
            data_dir, fileName = os.path.split(in_file)
            out_dir = data_dir.replace('../../../dataset/tumor_cls/tumor_dataset', '../../../dataset/tumor_cls/bet')
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            freeSurfer_bet_file(in_file, out_dir)
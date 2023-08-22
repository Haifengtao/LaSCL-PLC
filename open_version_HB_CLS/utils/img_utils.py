#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   img_utils.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/12 10:48   Bot Zhao      1.0         None
"""

# import lib
# import sklearn.cluster as cluster
# from skimage import measure as ms
# from scipy.ndimage.morphology import binary_fill_holes[
import numpy as np
import SimpleITK as sitk
import cv2 as cv
# from skimage import morphology
import matplotlib.pyplot as plt
from scipy import ndimage


def resize_image_itk(itkimage,
                     newSpacing,
                     resamplemethod=sitk.sitkNearestNeighbor):
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:
    :param newSpacing:such as [1,1,1]
    :param resamplemethod:
    :return:
    """
    newSpacing = np.array(newSpacing, float)
    originSpcaing = itkimage.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    factor = newSpacing / originSpcaing
    # print('factor:', factor)
    newSize = originSize / factor
    # print("new size1:", newSize)
    newSize = (newSize+0.5).astype(np.uint32)
    # print("new size2:", newSize)
    resampler.SetReferenceImage(itkimage)  # 将输出的大小、原点、间距和方向设置为itkimage
    resampler.SetOutputSpacing(newSpacing.tolist())  # 设置输出图像间距
    resampler.SetSize(newSize.tolist())  # 设置输出图像大小
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled


def resize_image_size(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int) #spacing肯定不能是整数
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled
# image_path = 'F:\lung_lobe\data\data\mask/'
# image_file = glob(image_path + '*.nrrd')
# for i in range(len(image_file)):
#     itkimage = sitk.ReadImage(image_file[i])
#     itkimgResampled = resize_image_itk(itkimage, (128,128,64),resamplemethod= sitk.sitkNearestNeighbor) #这里要注意：mask用最近邻插值，CT图像用线性插值
#     sitk.WriteImage(itkimgResampled,'F:\lung_lobe\data\data\mask_resize/' + image_file[i][len(image_path):])


def crop_img(img_arr, st, et, pad=0):
    """
    crop img by voxel index;
    :param img_arr: img array;
    :param st: start point;
    :param et: end point;
    :return: new image
    """
    temp = img_arr.copy()
    return temp[st[0]-pad:et[0]+1+pad, st[1]-pad:et[1]+1+pad, st[2]-pad:et[2]+1+pad]


def get_bbox(img_arr, pad=0):
    """
    get the bounding box of a img_array.
    :param img_arr: image_array
    :return: start point;end point;
    """
    index_info = np.nonzero(img_arr)

    if len(img_arr.shape) < 3:
        st = [np.max([np.min(index_info[0])-pad, 0]), np.max([np.min(index_info[1])-pad, 0])]
        et = [np.min([np.max(index_info[0])+pad, img_arr.shape[0]]), np.min([np.max(index_info[1])+pad, img_arr.shape[1]])]
    elif len(img_arr.shape) == 3:
        st = [np.max([np.min(index_info[0])-pad, 0]), np.max([np.min(index_info[1])-pad, 0]),
              np.max([np.min(index_info[2])-pad, 0])]
        et = [np.min([np.max(index_info[0])+pad, img_arr.shape[0]]), np.min([np.max(index_info[1])+pad, img_arr.shape[1]]),
              np.min([np.max(index_info[2])+pad, img_arr.shape[2]])]
    else:
        raise ValueError('ERROR: we just support 2d or 3d image!')
    return st, et


def normalize_0_1(img_arr, min_intensity=None, max_intensity=None, ):
    """
    Normalize the image to 0-1 .
    :param img_arr: image_array;
    :param min_intensity: float
    :param max_intensity: float
    :return: new_image
    """
    if min_intensity is None:
        min_intensity = np.min(img_arr)
    if max_intensity is None:
        max_intensity = np.max(img_arr)
    img_arr[img_arr >= max_intensity] = max_intensity
    img_arr[img_arr <= min_intensity] = min_intensity
    return ((img_arr - min_intensity) / (max_intensity+1)).astype(np.float32)


def normalize_uint8(img_arr, min_intensity=None, max_intensity=None, ):
    """
    Normalize the image to 0-1 .
    :param img_arr: image_array;
    :param min_intensity: float
    :param max_intensity: float
    :return: new_image
    """
    if min_intensity is None:
        min_intensity = np.min(img_arr)
    if max_intensity is None:
        max_intensity = np.max(img_arr)
    img_arr[img_arr > max_intensity] = max_intensity
    img_arr[img_arr < min_intensity] = min_intensity
    return (255*((img_arr - min_intensity) / max_intensity)).astype(np.uint8)


def resize_3d_arr(arr, new_size, order):
    """

    :param arr:
    :param new_size:
    :param order:  0,nearest ;1:双线性；3，cubic
    :return:
    """
    scales = np.array(new_size)/np.array(arr.shape)
    return ndimage.interpolation.zoom(arr, scales, order=order)



def rm_small_cc(mask_array, rate=0.3):
    """
    remove small object
    :param mask_array: input binary image
    :param rate:size rate
    :return:binary image
    """
    sitk_mask_img = sitk.GetImageFromArray(mask_array, )
    cc = sitk.ConnectedComponent(sitk_mask_img)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.SetGlobalDefaultNumberOfThreads(8)
    stats.Execute(cc, sitk_mask_img)

    max_label = 0
    maxsize = 0
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if maxsize < size:
            max_label = l
            maxsize = size

    not_remove = []
    for l in stats.GetLabels():
        size = stats.GetPhysicalSize(l)
        if size > maxsize * rate:
            not_remove.append(l)
    label_img = sitk.GetArrayFromImage(cc)
    out_mask = label_img.copy()
    out_mask[label_img != max_label] = 0

    for i in range(len(not_remove)):
        out_mask[label_img == not_remove[i]] = 1
    return out_mask


def fill_image_2d(image, filled_number=1):
    contours, _ = cv.findContours(image.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    num = len(contours)  # 轮廓的个数
    if num == 1:
        return image
    else:
        areas_contours = []
        fill_contours = []
        for contour in contours:
            area = cv.contourArea(contour)
            areas_contours.append(area)
        for idx, area in enumerate(areas_contours):
            if area <= np.max(areas_contours) * 0.3:
                # print("in...")
                fill_contours.append(contours[idx])
        cv.fillPoly(image, fill_contours, filled_number)
        return image


def fill_image_z(mask_arr, ):
    x, y, z = mask_arr.shape
    filled_img = np.zeros((x, y, z))
    for i in range(z):
        filled_img[:, :, i] = binary_fill_holes(mask_arr[:, :, i])
    return filled_img


def get_holes(mask_arr, ):
    x, y, z = mask_arr.shape
    filled_img = np.zeros((x, y, z))
    for i in range(z):
        temp = np.zeros((x, y))
        filled_2d = fill_image_2d(mask_arr[:, :, i].astype(np.uint8), filled_number=2)
        temp[filled_2d == 2] = 1
        filled_img[:, :, i] = temp
    return filled_img


def median_filter(mask_arr, radius=3):
    """
    median_filter for a 3d/2d image.
    :param mask_arr:
    :param radius:
    :return:
    """
    image = sitk.GetImageFromArray(mask_arr)
    sitk_median = sitk.MedianImageFilter()
    sitk_median.SetRadius(radius)
    sitk_median = sitk_median.Execute(image)
    median_array = sitk.GetArrayFromImage(sitk_median)
    return median_array


def dilation(data, kernel_size):
    """
    :param data:
    :param kernel_size:
    :return:
    """
    kernel = np.ones((kernel_size, kernel_size, kernel_size))
    enhance_data = morphology.dilation(data, kernel)  # dilation
    return enhance_data


def crop2raw(raw_shape, new_img, st, et):
    """
    crop the image roi to raw image.
    :param raw_shape: shape.
    :param new_img:
    :param st: start point.
    :param et: start point.
    :return: new image.
    """
    temp = np.zeros(raw_shape)
    temp[st[0]:et[0], st[1]:et[1], st[2]:et[2]] = new_img
    return temp


def rotate(image, angle, center=None, scale=1.0):
    """
    rotate 2d image any angle.
    :param image: numpy array;
    :param angle: 0~360;
    :param center:
    :param scale:
    :return: new image.
    """
    image = image.astype(np.float64)
    (h, w) = image.shape[:2]
    # if the center is None, initialize it as the center of the image
    if center is None:
        center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))
    return rotated


def vis_3d_img(img_array, bins=5, cmap='gray', save_dir = None):
    """
    visualize the 3d image in 3 dim.
    :param img_array:
    :param bins: how many slices to show; int
    :param cmap: colormap
    :return: None
    """
    assert len(img_array.shape) == 3, print("The image dim must be 3!")
    figure, ax = plt.subplots(3, bins, figsize=(20, 10))
    x, y, z = img_array.shape
    for i in range(bins):
        temp1 = ax[0][i].imshow(rotate(img_array[int(x / (bins + 1)) * (i + 1), ...].T, 180), cmap=cmap)
        temp2 = ax[1][i].imshow(rotate(img_array[:, int(y / (bins + 1)) * (i + 1), :].T, 180), cmap=cmap)
        temp3 = ax[2][i].imshow(rotate(img_array[:, :, int(z / (bins + 1)) * (i + 1)].T, 180), cmap=cmap)
        figure.colorbar(temp1, ax=ax[0][i])
        figure.colorbar(temp2, ax=ax[1][i])
        figure.colorbar(temp3, ax=ax[2][i])
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.show()


def crop_resize(img, img_array):
    z, y, x = img_array.shape
    # pad and resize
    if x <= 512 and y <= 512:
        resize_img = np.pad(img_array, (
            (0, 0), ((512 - y) // 2, (512 - y) - (512 - y) // 2), ((512 - x) // 2, (512 - x) - (512 - x) // 2),),
                            "constant", constant_values=((0, 0), (0, 0), (0, 0)))
    else:
        temp_img = resize_image_size(img, (512, 512, z), resamplemethod=sitk.sitkLinear)
        resize_img = sitk.GetArrayFromImage(temp_img)
    norm_arr = normalize_0_1(resize_img, min_intensity=0, max_intensity=3000)
    return norm_arr


def de_crop_resize(img_arr, img_ref):
    x, y, z = img_ref.GetSize()
    if x <= 512 and y <= 512:
        minx, miny = (512 - x) // 2, (512 - y) // 2
        maxx, maxy = ((512 - x) // 2) + x, ((512 - y) // 2) + y
        # pdb.set_trace()
        img_arr = img_arr[:, miny:maxy, minx:maxx]
    else:
        img_arr = sitk.GetImageFromArray(img_arr)
        temp_mask = resize_image_size(img_arr, (x, y, z))
        img_arr = sitk.GetArrayFromImage(temp_mask)
    return img_arr
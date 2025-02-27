from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
from PIL import Image
import cv2
from utils import *
from .mono_dataset_clinic import MonoDataset
# from mono_dataset import MonoDataset
from torch.utils.data import DataLoader

def undistort(img):
    # P = [[633.252071219191, -5.79020249611889, 566.149345290896],
    #      [0, 628.191459694915, 505.380462446338],
    #      [0, 0, 1.000]]  # clinic,1058*1007
    target_size =[224, 224]
    P = [[134.17, -5.79020249611889,119.91],
         [0, 139.70, 112.45],
         [0, 0, 1.000]]  # clinic,224*224

    K = [-0.384210779219478, 0.131041833595485, 0.000240872476213092, 0.00338294731237482, -0.0186712628446009]  # c3vd
    if isinstance(img, Image.Image):  # 如果是 PIL 图像
        # 将 PIL 图像转换为 NumPy 数组
        img = np.array(img)
    if img.shape[:2] != target_size:
        img = cv2.resize(img, target_size)
    img_dis = cv2.undistort(img, np.array(P), np.array(K))

    return img_dis



class CilincDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(CilincDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.598536929,   0,           0.535112803,     0],
                                 [0,             0.623824687,  0.501867391,    0],
                                 [0,             0,            1,         0],
                                 [0,             0,            0,         1]], dtype=np.float32)



    def check_depth(self):

        return True


    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        img_dis = undistort(color)
        color = Image.fromarray(img_dis)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


"""
Cilinicdataset :
frame1/pred178100.000000.jpg
frame2/pred178100.000000.jpg
...
"""
class ClinicRAWDataset(CilincDataset):
    def __init__(self, *args, **kwargs):
        super(ClinicRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str_0 = "{}{}".format(frame_index, self.img_ext)

        image_path = os.path.join(
            self.data_path, folder, "{}"
        )

        if os.path.exists(image_path.format(f_str_0)):
            image_path = image_path.format(f_str_0)
            # print(image_path)
        else:
            print(image_path.format(f_str_0))
            print("不存在该文件")
            raise Exception(f"Unknow file :{image_path}")
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str_0 = "{}.png".format(frame_index, self.img_ext)

        depth_path = os.path.join(
            self.data_path, folder+"_depth", "{}"
        )

        if os.path.exists(depth_path.format(f_str_0)):
            depth_path = depth_path.format(f_str_0)
            # print(image_path)

        else:
            print(depth_path.format(f_str_0))
            print("不存在该文件")
            raise Exception(f"Unknow file :{depth_path}")

        depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_gt = cv2.resize(depth_gt, [224, 224])
        depth_gt = (depth_gt / 255. / 256.) *20. * 1.3
        # print(depth_gt.shape)#(1080,1350)
        # print(depth_gt.max())
        # print(depth_gt.min())


        if do_flip:
            depth_gt = np.fliplr(depth_gt)#左右翻转

        return depth_gt

    def get_teacher(self, folder, frame_index, side, do_flip):
        f_str_0 = "{}.png".format(frame_index, self.img_ext)

        depth_path = os.path.join(
            self.data_path, folder+"_depth", "{}"
        )

        if os.path.exists(depth_path.format(f_str_0)):
            depth_path = depth_path.format(f_str_0)
            # print(image_path)

        else:
            print(depth_path.format(f_str_0))
            print("不存在该文件")
            raise Exception(f"Unknow file :{depth_path}")

        depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_gt = cv2.resize(depth_gt, [224,224])

        depth_gt = (depth_gt / 255. / 256.) *20. * 1.3

        if do_flip:
            depth_gt = np.fliplr(depth_gt)#左右翻转

        return depth_gt
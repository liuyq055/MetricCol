from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
from PIL import Image
import cv2
from utils import *
from .mono_dataset import MonoDataset
# from mono_dataset import MonoDataset
from torch.utils.data import DataLoader

def undistort(img):
    P = [[769.8074, -0.1412, 675.2264],
         [0, 769.7205, 548.9035],
         [0, 0, 1.000]]  # c3vd

    K = [-0.4542, 0.1792, -0.0013, 0.0007, -0.0285]  # c3vd
    if isinstance(img, Image.Image):  # 如果是 PIL 图像
        # 将 PIL 图像转换为 NumPy 数组
        img = np.array(img)
    img_dis = cv2.undistort(img, np.array(P), np.array(K))



    return img_dis



class C3VDDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(C3VDDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.570227706,   0,           0.500,     0],
                                 [0,             0.7127042,   0.50824,    0],
                                 [0,             0,            1,         0],
                                 [0,             0,            0,         1]],dtype=np.float32)
        #K[0:,]除以width,K[1:,]除以height
        #self.full_res_shape =  (1350. 1080)


    def check_depth(self):

        return True


    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder,frame_index, side))
        img_dis = undistort(color)
        color = Image.fromarray(img_dis)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


"""
C3VD dataset :
c1v1/0_color.png,c1v1/1_color.png....
c2v1/0_color.png...
...
"""
class C3VDRAWDataset(C3VDDataset):
    def __init__(self, *args, **kwargs):
        super(C3VDRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str_0 = "{}_color{}".format(frame_index, self.img_ext)
        f_str_4d = "{:04d}_color{}".format(frame_index, self.img_ext)



        image_path = os.path.join(
            self.data_path, folder, "{}"
        )

        if os.path.exists(image_path.format(f_str_0)):
            image_path = image_path.format(f_str_0)
            # print(image_path)
        elif os.path.exists(image_path.format(f_str_4d)):
            image_path = image_path.format(f_str_4d)
            # print(image_path)
        else:
            print(image_path.format(f_str_0))
            print(image_path.format(f_str_4d))
            print("不存在该文件")
            raise Exception(f"Unknow file :{image_path}")
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str_0 = "{}_depth.tiff".format(frame_index, self.img_ext)
        f_str_4d = "{:04d}_depth.tiff".format(frame_index, self.img_ext)

        depth_path = os.path.join(
            self.data_path, folder, "{}"
        )

        if os.path.exists(depth_path.format(f_str_0)):
            depth_path = depth_path.format(f_str_0)
            # print(image_path)
        elif os.path.exists(depth_path.format(f_str_4d)):
            depth_path = depth_path.format(f_str_4d)
            # print(image_path)
        else:
            print(depth_path.format(f_str_0))
            print(depth_path.format(f_str_4d))
            print("不存在该文件")
            raise Exception(f"Unknow file :{depth_path}")

        depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_gt = undistort(depth_gt)
        # cv2.imshow('dep', depth_gt)
        # cv2.waitKey(0)
        # cv2.destroyWindow()
        depth_gt = depth_gt/655.35
        # print(depth_gt.shape)#(1080,1350)
        # print(depth_gt.max())
        # print(depth_gt.min())


        if do_flip:
            depth_gt = np.fliplr(depth_gt)#左右翻转

        return depth_gt

    def get_teacher(self, folder, frame_index, side, do_flip):

        depth_path = os.path.join(
            self.data_path, folder, "{}"
        )
        depth_path0 = f"{self.data_path}/depth/{folder}_{frame_index}_pred.png"
        depth_path4 = f"{self.data_path}/depth/{folder}_{frame_index:04d}_pred.png"

        if os.path.exists(depth_path0):
            depth_path = depth_path0

        elif os.path.exists(depth_path4):
            depth_path = depth_path4

        else:
            print(depth_path0)
            print(depth_path4)
            print("不存在该文件")


        depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        depth_gt =(depth_gt/655.35 ) * 1.0


        if do_flip:
            depth_gt = np.fliplr(depth_gt)#左右翻转

        return depth_gt
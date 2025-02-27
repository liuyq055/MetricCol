from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
from PIL import ImageFile
import PIL.Image as pil
import torch
import torch.utils.data as data
from torchvision import transforms
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES=True

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class SimoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.png'):
        super(SimoDataset, self).__init__()

        self.K = np.array([[0.479166653, 0, 0.500, 0],
                           [0, 0.479166653, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.transforms.ColorJitter(self.brightness,self.contrast,self.saturation,self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the poses network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = inputs[(n, im, i - 1)]


        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
                # print(f"{n, im, i}", inputs[(n, im, i)].shape)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):


        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        img_pth = os.path.join(self.data_path, line[0])
        depth_pth = os.path.join(self.data_path, line[1])
        
        for i in self.frame_idxs:

            inputs[("color", i, -1)] = self.get_color(img_pth, do_flip)

        # adjusting intrinsics to match each d4v1_scale in the pyramid

        K = self.K.copy()

        inv_K = np.linalg.pinv(K)

        inputs[("K", 0)] = torch.from_numpy(K)

        inputs[("inv_K", 0)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness,self.contrast,self.saturation,self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(depth_pth, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))


        return inputs

    def get_color(self, img_pth,  do_flip):

        color = self.loader(img_pth)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


    def check_depth(self):
        return True

    def get_depth(self, depth_path, do_flip):

        depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_gt = (depth_gt/255./256.) *20.
        if do_flip:
            depth_gt = np.fliplr(depth_gt)#左右翻转
        return depth_gt

    def get_pose(self, folder, frame_index):
        raise NotImplementedError


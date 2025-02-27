from __future__ import absolute_import, division, print_function

import os
import torch
import models.encoders as encoders
import models.decoders as decoders
import numpy as np

from torch.utils.data import DataLoader
from utils.layers import transformation_from_parameters
from utils.utils import readlines
from options import MonodepthOptions
from datasets import C3VDRAWDataset
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import evo.core.lie_algebra as lie
from evo.core import trajectory
from evo.tools import plot, file_interface, log
from evo.core import trajectory,units
from evo.core.units import Unit
import copy
import logging
from evo.core import metrics
import pprint

units.METER_SCALE_FACTORS[Unit.millimeters]=1


def evaluate(opt, scene):
    """Evaluate odometry on the SCARED dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    filenames = readlines(
        os.path.join(os.path.dirname(__file__), "splits", "c3vd",
                     f"test_files_{scene}.txt"))

    dataset = C3VDRAWDataset(opt.data_path, filenames, opt.height, opt.width,
                               [0, 1], 4, is_train=False)
    # print(len(dataset))
    dataloader = DataLoader(dataset, 1, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")
    intrinsics_decoder_path = os.path.join(opt.load_weights_folder, "intrinsics_head.pth")

    pose_encoder = encoders.ResnetEncoder(opt.num_layers, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))

    pose_decoder = decoders.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    if opt.learn_intrinsics:
        intrinsics_decoder = decoders.IntrinsicsHead(pose_encoder.num_ch_enc)
        intrinsics_decoder.load_state_dict(torch.load(intrinsics_decoder_path))
        intrinsics_decoder.cuda()
        intrinsics_decoder.eval()

    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    pred_poses = []
    pred_intrinsics = []

    print(f"-> Computing poses predictions in {scene}")

    opt.frame_ids = [0, 1]  # poses network only takes two frames as input

    with torch.no_grad():
        for inputs in dataloader:
            for key, ipt in inputs.items():
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].cuda()
                    # print()
                else:
                    inputs[key] = inputs[key]
                #     print(inputs[key])
                # print(key)


            all_color_aug = torch.cat([inputs[("color", 1, 0)], inputs[("color", 0, 0)]], 1)

            features = [pose_encoder(all_color_aug)]
            axisangle, translation, intermediate_feature = pose_decoder(features)

            pred_poses.append(
                    transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=False).cpu().numpy())

            if opt.learn_intrinsics:
                cam_K = intrinsics_decoder(
                        intermediate_feature, opt.width, opt.height)
                pred_intrinsics.append(cam_K[:,:3,:3].cpu().numpy())

    pred_poses = np.concatenate(pred_poses)#[n,4,4]

    #这一步将列表 pred_poses 中的所有矩阵沿第一个轴（时间维度）拼接，得到一个多维的 NumPy 数组 pred_poses，可以方便地保存到文件。
    if opt.learn_intrinsics:
        pred_intrinsics = np.concatenate(pred_intrinsics)

    save_dir = f'results/{opt.eval_split}/poses/{scene}'
    os.makedirs(save_dir,exist_ok=True)



    np.save(f'{save_dir}/{scene}_pred_poses.npy', pred_poses)

    if opt.learn_intrinsics:
        fx_mean, fx_std = np.mean(pred_intrinsics[:,0,0]) / opt.width, np.std(pred_intrinsics[:,0,0]) / opt.width
        fy_mean, fy_std = np.mean(pred_intrinsics[:,1,1]) / opt.height, np.std(pred_intrinsics[:,1,1]) / opt.height
        cx_mean, cx_std = np.mean(pred_intrinsics[:,0,2]) / opt.width, np.std(pred_intrinsics[:,0,2]) / opt.width
        cy_mean, cy_std = np.mean(pred_intrinsics[:,1,2]) / opt.height, np.std(pred_intrinsics[:,1,2]) / opt.height
        output_text = (

                "\n   fx: {:0.4f}, std: {:0.4f}\n".format(fx_mean, fx_std) +
                "\n   fy: {:0.4f}, std: {:0.4f}\n".format(fy_mean, fy_std) +
                "\n   cx: {:0.4f}, std: {:0.4f}\n".format(cx_mean, cx_std) +
                "\n   cy: {:0.4f}, std: {:0.4f}\n".format(cy_mean, cy_std)
        )

        print(output_text)
        with open(f"{save_dir}/{scene}_instrinsics.txt", "w") as f:
            # 将 ATE 和 RE 的结果写入文件
            f.write(output_text)

    # 提示完成写入
    print("Results have been written to 'trajectory_and_rotation_error.txt'")




if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()
    # scene = 's3v2'
    test_scene = ["s3v2", "t3v1", "t4v1","d4v1","c4v1"]
    for scene in test_scene:
        evaluate(opt, scene=scene)

    model_name = opt.load_weights_folder.replace('/', '_')
    with open(f'results/{opt.eval_split}/poses/{model_name}.txt', 'w') as f:
        f.write(f'loading weights {opt.load_weights_folder}')


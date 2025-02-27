from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
import scipy.stats as st
import re
from utils.layers import disp_to_depth
from utils.utils import readlines, compute_errors,colorize
from options import MonodepthOptions
import datasets
import models.encoders as encoders
import models.decoders as decoders
import models.endodac as endodac

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def render_depth(disp):
    disp = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp = disp.astype(np.uint8)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_INFERNO)
    return disp_color

def build(opt):
    model = endodac.endodac(
        backbone_size="base", r=4, lora_type=opt.lora_type,
        image_shape=(224, 280),
        residual_block_indexes=opt.residual_block_indexes,
        include_cls_token=opt.include_cls_token, multi_metric=opt.multi_metric)
    pth_path = f'{opt.pretrained_path}/{opt.load}'
    assert isinstance(pth_path, str), "pretrained_resource must be a string"

    state_dict = torch.load(pth_path)
    mapped_state_dict = {}
    for pretrained_name, weight in state_dict.items():

        new_name = re.sub(r'pretrained\.', 'encoder.', pretrained_name)
        # new_name = re.sub(r'depth_head\.',
        #                   'depth_head.', new_name)

        if new_name in model.state_dict():
            mapped_state_dict[new_name] = weight

    model.load_state_dict(mapped_state_dict, strict=False)

    return model
def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 100

    # assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
    #     "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:
        if not opt.model_type == 'depthanything':
            opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
            assert os.path.isdir(opt.load_weights_folder), \
                "Cannot find a folder at {}".format(opt.load_weights_folder)

            print("-> Loading models from {}".format(opt.load_weights_folder))
        else:
            print("Evaluating Depth Anything model")

        if opt.model_type == 'endodac':
            depther_path = os.path.join(opt.load_weights_folder, "depth_model.pth")
            depther_dict = torch.load(depther_path)
            print(f'depth model loading {depther_path}')


        if opt.eval_split == 'endovis':
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
            dataset = datasets.SCAREDRAWDataset(opt.data_path, filenames,
                                            opt.height, opt.width,
                                            [0], 4, is_train=False)

        elif opt.eval_split == 'c3vd':
            fpath = os.path.join(os.path.dirname(__file__), "splits", opt.eval_split, "{}_files.txt")
            test_filenames = readlines(fpath.format("test"))
            img_ext = '.png'
            dataset = datasets.C3VDRAWDataset(opt.data_path, test_filenames, opt.height, opt.width,
                                                [0], 4, is_train=False, img_ext=img_ext)
            MAX_DEPTH = 100

        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        if opt.model_type == 'endodac':

            depther = build(opt)
            model_dict = depther.state_dict()

            depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})

            depther.cuda()
            depther.eval()
        elif opt.model_type == 'endours':

            depther = build(opt)
            model_dict = depther.state_dict()

            depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})

            depther.cuda()
            depther.eval()


    else:
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)
        if opt.eval_split == 'endovis':
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
            dataset = datasets.SCAREDRAWDataset(opt.data_path, filenames,
                                            opt.height, opt.width,
                                            [0], 4, is_train=False)
        elif opt.eval_split == 'hamlyn':
            dataset = datasets.HamlynDataset(opt.data_path, opt.height, opt.width,
                                                [0], 4, is_train=False)
        elif opt.eval_split == 'c3vd':
            fpath = os.path.join(os.path.dirname(__file__), "splits", opt.split, "{}_files.txt")
            test_filenames = readlines(fpath.format("test"))
            img_ext = '.png'
            dataset = datasets.C3VDRAWDataset(opt.data_path, test_filenames, opt.height, opt.width,
                                              [0], 4, is_train=False, img_ext=img_ext)
            MAX_DEPTH = 100


        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        
    if opt.visualize_depth:
        vis_dir = os.path.join('results',opt.eval_split, "vis_depth")
        os.makedirs(vis_dir, exist_ok=True)

    inference_times = []
    sequences = []
    keyframes = []
    frame_ids = []
    pred_disps =[]
    
    errors = []
    ratios = []
    print("-> Computing predictions with size {}x{}".format(
        opt.width, opt.height))

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            input_color = data[("color", 0, 0)].cuda()
            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            if opt.ext_disp_to_eval is None:
                time_start = time.time()
                output = depther(input_color)
                inference_time = time.time() - time_start
                if opt.model_type == 'endodac' or opt.model_type == 'afsfm':
                    output_disp = output[("depth", 0)]
                    pred_disp, _ = disp_to_depth(output_disp, opt.min_depth, opt.max_depth)

                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disp = pred_disp[0]
            else:
                pred_disp = pred_disps[i]
                inference_time = 1
            inference_times.append(inference_time)
            pred_disps.append(pred_disp)



            if opt.eval_split == 'hamlyn' or opt.eval_split == 'c3vd':
                gt_depth = data["depth_gt"].squeeze(0).squeeze(0).numpy()
                sequence = str(data['sequence'][0])
                keyframe = str(data['keyframe'][0])

            gt_height, gt_width = gt_depth.shape[:2]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1/pred_disp
            print(f"ori_depth max:{pred_depth.max()}, min:{pred_depth.min()} ")
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= opt.pred_depth_scale_factor
            if not opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                if not np.isnan(ratio).all():
                    ratios.append(ratio)
                pred_depth *= ratio

            print(f'gt_depth max {gt_depth.max()}, min:{gt_depth.min()}')
            print(f"scale_depth max:{pred_depth.max()}, min:{pred_depth.min()} ")

            if opt.visualize_depth:
                depth = np.reshape(pred_depth, (gt_height, gt_width))
                file_name = os.path.join(vis_dir, sequence + "_" + keyframe + ".tiff")
                img = Image.fromarray((depth * 655.35).astype(np.uint16), mode="I;16")
                img.save(file_name)
                # vis_pred_depth = render_depth(pred_disp)
                vis_pred_depth = colorize(np.reshape(pred_depth, (gt_height, gt_width)), MIN_DEPTH, MAX_DEPTH)
                vis_file_name = os.path.join(vis_dir, sequence + "_" +  keyframe + "_color.png")
                Image.fromarray(vis_pred_depth).save(vis_file_name)


            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
            error = compute_errors(gt_depth, pred_depth)
            if not np.isnan(error).all():
                errors.append(error)

        pred_disps = np.concatenate(pred_disps)
        if opt.save_pred_disps:
            output_path = os.path.join(
                opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
            print("-> Saving predicted disparities to ", output_path)
            np.save(output_path, pred_disps)

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    errors = np.array(errors)
    mean_errors = np.mean(errors, axis=0)
    cls = []
    for i in range(len(mean_errors)):
        cl = st.t.interval(confidence=0.95, df=len(errors)-1, loc=mean_errors[i], scale=st.sem(errors[:,i]))
        cls.append(cl[0])
        cls.append(cl[1])
    cls = np.array(cls)

    print("\n       " + ("{:>11}      | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print("mean:" + ("&{: 12.4f}      " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("cls: " + ("& [{: 6.4f}, {: 6.4f}] " * 7).format(*cls.tolist()) + "\\\\")
    print("average inference time: {:0.1f} ms".format(np.mean(np.array(inference_times))*1000))
    print("\n-> Done!")
    with open(f'results/{opt.eval_split}/errors.txt', 'w') as f:
        f.write(f'loading depth model:{opt.load_weights_folder}\n')
        if not opt.disable_median_scaling:
            f.write(" Scaling ratios | med: {:0.4f} | std: {:0.4f}\n".format(med, np.std(ratios / med)))
        f.write("\n       " + ("{:>11}      | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        f.write("\nmean:" + ("&{: 12.4f}      " * 7).format(*mean_errors.tolist()) + "\\\\")
        f.write("\ncls: " + ("& [{: 6.3f}, {: 6.3f}] " * 7).format(*cls.tolist()) + "\\\\")
        f.write("\naverage inference time: {:0.1f} ms".format(np.mean(np.array(inference_times))*1000))



if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())

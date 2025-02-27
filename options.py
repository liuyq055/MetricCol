from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

def str2bool(v):
     if isinstance(v, bool):
          return v
     if v.lower() in ('yes', 'true', 't', 'y', '1'):
          return True
     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
          return False
     else:
          raise argparse.ArgumentTypeError('Boolean value expected.')

class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default="../dataset/c3vd")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default='logs')


        # Model options
        self.parser.add_argument("--pretrained_path",
                                 type=str,
                                 help="pretrained models path",
                                 default=os.path.join(file_dir, "pretrained_model"))
        self.parser.add_argument("--lora_type",
                                 type=str,
                                 help="which lora type use for the model",
                                 choices=["lora", "dvlora", "none"],
                                 default="dvlora")
        self.parser.add_argument("--lora_rank",
                                 type=int,
                                 help="the rank of lora",
                                 default=4)
        self.parser.add_argument("--warm_up_step",
                                 type=int,
                                 help="warm up step",
                                 default=20000)
        self.parser.add_argument("--residual_block_indexes",
                                 nargs="*",
                                 type=int,
                                 help="indexes for residual blocks in vitendodepth encoder",
                                 default=[2,5,8,11])
        self.parser.add_argument("--include_cls_token",
                                 type=str2bool,
                                 help="includes the cls token in the transformer blocks",
                                 default=True)
        self.parser.add_argument("--learn_intrinsics",
                                 type=str2bool,
                                 help="learn the camera intrinsics with a seperate decoder",
                                 default=False)


        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="endours")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["endovis","c3vd"],
                                 default="c3vd")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet Metric_Layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="c3vd",
                                 choices=["endovis","c3vd"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=256)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=320)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--position_smoothness",
                                 type=float,
                                 help="registration smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--transform_constraint",
                                 type=float,
                                 help="transform constraint weight",
                                 default=0.001)
        self.parser.add_argument("--transform_smoothness",
                                 type=float,
                                 help="transform smoothness weight",
                                 default=0.01)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0,1,2,3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.01)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=1.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")

        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=1)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=50)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=10)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the poses network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=8)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["position_encoder", "position"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=400)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--model_type",
                                 type=str,
                                 help="which training split to use",
                                 choices=["endodac", "afsfm"],
                                 default="endodac")
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 type=str2bool,
                                 default=False)
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=100.)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="c3vd",
                                 choices=[
                                    "clinic", "c3vd", "endovis"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 default="store_true")
        self.parser.add_argument("--visualize_depth",
                                 help="if set saves visualized depth map",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

        # EVALUATION options
        self.parser.add_argument("--save_recon",
                                 help="if set saves reconstruction files",
                                 action="store_true")

        self.parser.add_argument('--config',
                            type=str, help='Path to configuration file')

        #MODIFY options
        self.parser.add_argument("--exp_name",
                                 type=str,
                                 help="exp directory",
                                 default='17(vit-small)')
        self.parser.add_argument("--metric_depth",
                                 type=str2bool,
                                 help="whether use metric depth head",
                                 default=True)
        self.parser.add_argument("--use_depth",
                                 help="if set, uses depth map for training",
                                 type=str2bool,
                                 default=True)
        self.parser.add_argument("--multi_metric",
                                 help="if set, uses multi metric depth map",
                                 type=str2bool,
                                 default=False)
        self.parser.add_argument("--use_3d",
                                 help="if set, uses 3d loss for training",
                                 type=str2bool,
                                 default=False)

        self.parser.add_argument("--load",
                                 help="which pth to use",
                                 type=str,
                                 default='depth_anything_vitb14.pth')
        self.parser.add_argument("--different_lr",
                                 help="if set,using different lr in depth model",
                                 type=str2bool,
                                 default=False)
        self.parser.add_argument("--dramatic",
                                 help="if set,using dramatic weight for depth loss",
                                 type=str2bool,
                                 default=False)
        self.parser.add_argument("--pose_label",
                                 help="if set,using depth label for pose",
                                 type=str2bool,
                                 default=False)
        self.parser.add_argument("--use_teacher",
                                 help="if set,using teacher model to generate dpeth",
                                 type=str2bool,
                                 default=True)
        self.parser.add_argument("--pc_label",
                                 help="if set,using depth for pc generate",
                                 type=str2bool,
                                 default=False)
        self.parser.add_argument("--use_confi",
                                 help="if set,using depth for pc generate",
                                 type=str2bool,
                                 default=False)
        self.parser.add_argument("--weight_depth",
                                 type=float,
                                 help="transform smoothness weight",
                                 default=0.001)
        self.parser.add_argument("--weight_pc",
                                 type=float,
                                 help="pc  weight",
                                 default=0.001)
        self.parser.add_argument("--colondepth_pretrain",
                                 type=str,
                                 help="transform smoothness weight",
                                 default=None)
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options

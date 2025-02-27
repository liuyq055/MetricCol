import os
import torch
import torch.nn as nn
import models.backbones as backbones
from models.backbones.mylora import Linear as LoraLinear
from models.backbones.mylora import DVLinear as DVLinear
from .layers import HeadDepth
from .layers import mark_only_part_as_trainable,_make_scratch, _make_fusion_block

def compare_and_save_model_params(model1, model2, file_path="check/small/model_params_comparison.txt"):
    """
    Compare two model's parameters and save their details and differences into a txt file.

    Args:
    - model1 (torch.nn.Module): First model to compare.
    - model2 (torch.nn.Module): Second model to compare.
    - file_path (str): Path to the output text file.
    """
    with open(file_path, "w") as f:
        # Write Model 1 parameters
        f.write("Model 1 Parameters:\n")
        for name, param in model1.items():
            f.write(f"Name: {name}, Param: {param}\n")

        f.write("\nModel 2 Parameters:\n")
        # Write Model 2 parameters
        for name, param in model2.state_dict().items():
            f.write(f"Name: {name}, Param: {param}\n")

        f.write("\nDifferences Between Models:\n")
        # Compare Model 1 and Model 2 parameters
        for name, param1 in model1.items():

            if name in model2.state_dict():
                param2 = model2.state_dict()[name]
                if param1.shape != param2.shape:
                    f.write(
                        f"Shape Mismatch - Name: {name}, Model 1 Shape: {list(param1.shape)}, Model 2 Shape: {list(param2.shape)}\n")
                elif not torch.equal(param1, param2):
                    f.write(f"Value Difference - Name: {name}\n")
            else:
                f.write(f"Parameter Missing in Model 2: {name}\n")

        # Check for parameters in model2 not in model1
        for name in model2.state_dict():
            model1_name = [name for name, _ in model1.items()]
            # print([name for name in model1.items()])
            if name not in model1_name:
                f.write(f"Parameter Missing in Model 1: {name}\n")
            # if name in model1_name:
            #     f.write(f"Parameter Matched in Model 1: {name}\n")
class DPTHead(nn.Module):
    def __init__(self, in_channels, features=128, use_bn=False, out_channels=[96, 192, 384, 768], use_clstoken=False, multi_metric = False):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken
        self.multi_metric = multi_metric
        layer_names = ('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1')
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.conv_depth_1 = HeadDepth(features)
        self.conv_depth_2 = HeadDepth(features)
        self.conv_depth_3 = HeadDepth(features)
        self.conv_depth_4 = HeadDepth(features)
        
        self.sigmoid = nn.Sigmoid()
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        outputs = {}

        if self.multi_metric:
            outputs[("depth", 0)] = self.conv_depth_1(path_1)
        else:

            outputs[("depth", 3)] = self.sigmoid(self.conv_depth_4(path_4))
            outputs[("depth", 2)] = self.sigmoid(self.conv_depth_3(path_3))
            outputs[("depth", 1)] = self.sigmoid(self.conv_depth_2(path_2))
            outputs[("depth", 0)] = self.sigmoid(self.conv_depth_1(path_1))

        return outputs
    
class endodac(nn.Module):
    """Applies low-rank adaptation to a ViT model's image encoder.

    Args:
        backbone_size: size of pretrained Dinov2 choice from: "small", "base", "large", "giant"
        r: rank of LoRA
        image_shape: input image shape, h,w need to be multiplier of 14, default:(224,280)
        lora_layer: which layer we apply LoRA.
    """

    def __init__(self, 
                 backbone_size = "base", 
                 r=4, 
                 image_shape=(224,280), 
                 lora_type="lora",
                 residual_block_indexes=[],
                 include_cls_token=True,
                 multi_metric=False,
                 use_cls_token=False,
                 use_bn=False):
        super(endodac, self).__init__()

        assert r > 0
        self.r = r
        self.backbone_size = backbone_size
        self.backbone = {
            "small": backbones.vits.vit_small(residual_block_indexes=residual_block_indexes,
                                              include_cls_token=include_cls_token),
            "base": backbones.vits.vit_base(residual_block_indexes=residual_block_indexes,
                                            include_cls_token=include_cls_token),
        }
        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
        }
        self.intermediate_layers = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
        }
        self.embedding_dims = {
            "small": 384,
            "base": 768,
        }
        self.depth_head_features = {
            "small": 64,
            "base": 128,
        }
        self.depth_head_out_channels = {
            "small": [48, 96, 192, 384],
            "base": [96, 192, 384, 768],
        }
        self.backbone_arch = self.backbone_archs[self.backbone_size]
        self.embedding_dim = self.embedding_dims[self.backbone_size]
        self.depth_head_feature = self.depth_head_features[self.backbone_size]
        self.depth_head_out_channel = self.depth_head_out_channels[self.backbone_size]
        encoder = self.backbone[self.backbone_size]

        self.image_shape = image_shape
        
        if lora_type != "none":
            for t_layer_i, blk in enumerate(encoder.blocks):
                mlp_in_features = blk.mlp.fc1.in_features
                mlp_hidden_features = blk.mlp.fc1.out_features
                mlp_out_features = blk.mlp.fc2.out_features
                if lora_type == "dvlora":
                    blk.mlp.fc1 = DVLinear(mlp_in_features, mlp_hidden_features, r=self.r, lora_alpha=self.r)
                    blk.mlp.fc2 = DVLinear(mlp_hidden_features, mlp_out_features, r=self.r, lora_alpha=self.r)
                elif lora_type == "lora":
                    blk.mlp.fc1 = LoraLinear(mlp_in_features, mlp_hidden_features, r=self.r)
                    blk.mlp.fc2 = LoraLinear(mlp_hidden_features, mlp_out_features, r=self.r)
            
        self.encoder = encoder
        self.depth_head = DPTHead(self.embedding_dim, self.depth_head_feature, use_bn,
                                  out_channels=self.depth_head_out_channel, use_clstoken=use_cls_token,  multi_metric= multi_metric)
        
        with open('check/small/weixunlian.txt', "w") as f:
            # Write Model 1 parameters
            f.write("Model 1 Parameters:\n")
            for name, param in self.state_dict().items():
                f.write(f"Name: {name}, Param: {param}\n")

        # pretrained_path = os.path.join( "pretrained_model", "depth_anything_{}.pth".format(self.backbone_arch))
        # pretrained_dict = torch.load(pretrained_path)
        # self.load_state_dict(pretrained_dict, strict=False)
        #
        #
        # with open('check/small/pretrain.txt', "w") as f:
        #     # Write Model 1 parameters
        #     f.write("Model 1 Parameters:\n")
        #     for name, param in pretrained_dict.items():
        #         f.write(f"Name: {name}, Param: {param}\n")
        # print("load pretrained weight from {}\n".format(pretrained_path))
        #
        # mark_only_part_as_trainable(self.encoder)
        # mark_only_part_as_trainable(self.depth_head)
        #
        # with open('check/small/jiazai.txt', "w") as f:
        #     # Write Model 1 parameters
        #     f.write("Model 1 Parameters:\n")
        #     for name, param in self.state_dict().items():
        #         f.write(f"Name: {name}, Param: {param}\n")
        # compare_and_save_model_params(pretrained_dict, self)

    def forward(self, pixel_values):
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=self.image_shape, mode="bilinear", align_corners=True)
        h, w = pixel_values.shape[-2:]
        
        features = self.encoder.get_intermediate_layers(pixel_values, 4, return_class_token=True)
        patch_h, patch_w = h // 14, w // 14

        disp = self.depth_head(features, patch_h, patch_w)

        return disp


class metric_endo(nn.Module):
    def __init__(self,core, layer_names=('depth_1', 'l4_rn', 'r4', 'r3', 'r2', 'r1')):
        super().__init__()
        self.core = core
        self.output_channels = None
        self.core_out = {}
        self.handles = []
        self.trainable = True

        self.layer_names = layer_names
        self.attach_hooks(core)


    def attach_hooks(self, midas):
        if len(self.handles) > 0:
            self.remove_hooks()


        if "depth_1" in self.layer_names:
            self.handles.append(list(midas.depth_head.conv_depth_1.children())[0][2].register_forward_hook(
                get_activation("depth_1", self.core_out)))
        if "r4" in self.layer_names:
            self.handles.append(midas.depth_head.scratch.refinenet4.register_forward_hook(
                get_activation("r4", self.core_out)))
        if "r3" in self.layer_names:
            self.handles.append(midas.depth_head.scratch.refinenet3.register_forward_hook(
                get_activation("r3", self.core_out)))
        if "r2" in self.layer_names:
            self.handles.append(midas.depth_head.scratch.refinenet2.register_forward_hook(
                get_activation("r2", self.core_out)))
        if "r1" in self.layer_names:
            self.handles.append(midas.depth_head.scratch.refinenet1.register_forward_hook(
                get_activation("r1", self.core_out)))
        if "l4_rn" in self.layer_names:
            self.handles.append(midas.depth_head.scratch.layer4_rn.register_forward_hook(
                get_activation("l4_rn", self.core_out)))

        return self
    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        return self

    def __del__(self):
        self.remove_hooks()

    def forward(self, x):

        with torch.set_grad_enabled(self.trainable):
            disp= self.core(x)

        out = [self.core_out[k] for k in self.layer_names]
        # for k in self.layer_names:
        #     print(f'{k}:', self.core_out[k].shape)

        return disp, out

def get_activation(name, bank):
    def hook(model, input, output):
        bank[name] = output

    return hook
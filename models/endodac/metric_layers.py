from models.endodac.endodac import endodac
import torch
from models.endodac.Metric_Layers.attractor import AttractorLayer, AttractorLayerUnnormed
from models.endodac.Metric_Layers.dist_layers import ConditionalLogBinomial
from models.endodac.Metric_Layers.localbins_layers import (Projector, SeedBinRegressorUnnormed,SeedBinRegressor)
import torch.nn as nn
import itertools

class ColonDepth(nn.Module):
    def __init__(self, core, multi_metric=False, n_bins=64, bin_centers_type="softplus", bin_embedding_dim=128, min_depth=1e-3,
                 max_depth=1.,
                 n_attractors=[16, 8, 4, 1], attractor_alpha=1000, attractor_gamma=2, attractor_kind='mean',
                 attractor_type='inv', min_temp=0.0212, max_temp=50):
        super().__init__()

        self.core = core
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.min_temp = min_temp
        self.bin_centers_type = bin_centers_type
        self.multi_metric =multi_metric



        N_MIDAS_OUT = 32
        output_channels = [self.core.core.depth_head_feature] *5

        btlnck_features = output_channels[0]
        num_out_features = output_channels[1:]

        self.conv2 = nn.Conv2d(btlnck_features, btlnck_features,
                               kernel_size=1, stride=1, padding=0)  # btlnck conv

        if bin_centers_type == "normed":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayer
        elif bin_centers_type == "softplus":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid1":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid2":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayer
        else:
            raise ValueError(
                "bin_centers_type should be one of 'normed', 'softplus', 'hybrid1', 'hybrid2'")
        self.seed_bin_regressor = SeedBinRegressorLayer(
            btlnck_features, n_bins=n_bins, min_depth=min_depth, max_depth=max_depth)
        self.seed_projector = Projector(btlnck_features, bin_embedding_dim)
        self.projectors = nn.ModuleList([
            Projector(num_out, bin_embedding_dim)
            for num_out in num_out_features
        ])
        self.attractors = nn.ModuleList([
            Attractor(bin_embedding_dim, n_bins, n_attractors=n_attractors[i], min_depth=min_depth, max_depth=max_depth,
                      alpha=attractor_alpha, gamma=attractor_gamma, kind=attractor_kind, attractor_type=attractor_type)
            for i in range(len(num_out_features))
        ])

        last_in = N_MIDAS_OUT + 1  # +1 for relative depth

        # use log binomial instead of softmax
        self.conditional_log_binomial = ConditionalLogBinomial(
            last_in, bin_embedding_dim, n_classes=n_bins, min_temp=min_temp, max_temp=max_temp)

        # self.conditional_log_binomial_prev = ConditionalLogBinomial(
        #     65, bin_embedding_dim, n_classes=n_bins, min_temp=min_temp, max_temp=max_temp)


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
            return_final_centers (bool, optional): Whether to return the final bin centers. Defaults to False.
            denorm (bool, optional): Whether to denormalize the input image. This reverses ImageNet normalization as midas normalization is different. Defaults to False.
            return_probs (bool, optional): Whether to return the output probability distribution. Defaults to False.

        Returns:
            dict: Dictionary containing the following keys:
                - rel_depth (torch.Tensor): Relative depth map of shape (B, H, W)
                - metric_depth (torch.Tensor): Metric depth map of shape (B, 1, H, W)
                - bin_centers (torch.Tensor): Bin centers of shape (B, n_bins). Present only if return_final_centers is True
                - probs (torch.Tensor): Output probability distribution of shape (B, n_bins, H, W). Present only if return_probs is True

        """
        depth = {}
        b, c, h, w = x.shape

        self.orig_input_width = w
        self.orig_input_height = h
        disp, out = self.core(x)

        outconv_activation = out[0]
        btlnck = out[1]
        x_blocks = out[2:]

        #btlnck
        x_d0 = self.conv2(btlnck)
        x = x_d0
        _, seed_b_centers = self.seed_bin_regressor(x)

        if self.bin_centers_type == 'normed' or self.bin_centers_type == 'hybrid2':
            b_prev = (seed_b_centers - self.min_depth) / \
                     (self.max_depth - self.min_depth)
        else:
            b_prev = seed_b_centers

        prev_b_embedding = self.seed_projector(x)

        centers = {}
        embedding = {}

        for projector, attractor, x in zip(self.projectors, self.attractors, x_blocks):
            b_embedding = projector(x)
            b, b_centers = attractor(
                b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()


        last = outconv_activation
        rel_cond = disp[("depth", 0)]

        # print(f"rel cond max:{rel_cond.max()},rel cond min:{rel_cond.min()} ")

        rel_cond = nn.functional.interpolate(
            rel_cond, size=last.shape[2:], mode='bilinear', align_corners=True)

        last = torch.cat([last, rel_cond], dim=1)

        b_embedding = nn.functional.interpolate(
            b_embedding, last.shape[-2:], mode='bilinear', align_corners=True)

        x = self.conditional_log_binomial(last, b_embedding)
        # print(f"x max:{x.max()}")

        b_centers = nn.functional.interpolate(
            b_centers, x.shape[-2:], mode='bilinear', align_corners=True)
        # print(f"b_centers max:{b_centers.max()}")
        # print(f"b_centers shape after interpolate: {b_centers.shape}")
        depth[("depth", 0)] = torch.sum(x * b_centers, dim=1, keepdim=True)

        # Structure output dict

        depth[('probs', 0)] = x


        depth[("depth", 0)] = torch.sum(x * b_centers, dim=1, keepdim=True)

        return depth

    def get_lr_params(self, lr):
        """
        Learning rate configuration for different layers of the model
        Args:
            lr (float) : Base learning rate
        Returns:
            list : list of parameters to optimize and their learning rates, in the format required by torch optimizers.
        """
        param_conf = []
        if self.train_midas:
            print("Training depth anything...")
            if self.encoder_lr_factor > 0:
                param_conf.append({'params': self.core.get_enc_params_except_rel_pos(
                ), 'lr': lr / self.encoder_lr_factor})#训练编码器

            if self.pos_enc_lr_factor > 0:
                param_conf.append(
                    {'params': self.core.get_rel_pos_params(), 'lr': lr / self.pos_enc_lr_factor})

            # midas_params = self.core.core.scratch.parameters()
            midas_params = self.core.core.depth_head.parameters()#解码器
            midas_lr_factor = self.midas_lr_factor
            param_conf.append(
                {'params': midas_params, 'lr': lr / midas_lr_factor})

        remaining_modules = []
        for name, child in self.named_children():
            if name != 'core':
                remaining_modules.append(child)
        remaining_params = itertools.chain(
            *[child.parameters() for child in remaining_modules])

        param_conf.append({'params': remaining_params, 'lr': lr})

        return param_conf



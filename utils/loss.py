import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from utils.metrics import cd, emd, fscore
#
#
# def calc_dcd(x, gt, alpha=1000, n_lambda=1, return_raw=False, non_reg=False):
#     x = x.float().permute(0, 2, 1)
#     gt = gt.float().permute(0, 2, 1)
#     batch_size, n_x, _ = x.shape
#     batch_size, n_gt, _ = gt.shape
#     assert x.shape[0] == gt.shape[0]
#
#     if non_reg:
#         frac_12 = max(1, n_x / n_gt)
#         frac_21 = max(1, n_gt / n_x)
#     else:
#         frac_12 = n_x / n_gt
#         frac_21 = n_gt / n_x
#
#     cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
#     # print(cd_p,cd_t)
#     # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
#     # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
#     # dist2 and idx2: vice versa
#     exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)
#
#     count1 = torch.zeros_like(idx2)
#     count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
#     weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
#     weight1 = (weight1 + 1e-6) ** (-1) * frac_21
#     loss1 = (1 - exp_dist1 * weight1).mean(dim=1)
#     # print(loss1)
#
#     count2 = torch.zeros_like(idx1)
#     count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
#     weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
#     weight2 = (weight2 + 1e-6) ** (-1) * frac_12
#     loss2 = (1 - exp_dist2 * weight2).mean(dim=1)
#     # print(loss2)
#
#     loss = (loss1 + loss2) / 2
#
#     res = [loss, cd_p, cd_t]
#     if return_raw:
#         res.extend([dist1, dist2, idx1, idx2])
#
#     return res
#
# def calc_cd(output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False):
#     # cham_loss = dist_chamfer_3D.chamfer_3DDist()
#     cham_loss = cd()
#     dist1, dist2, idx1, idx2 = cham_loss(gt, output)
#     cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
#     cd_t = (dist1.mean(1) + dist2.mean(1))
#
#     if separate:
#         res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
#                torch.cat([dist1.mean(1).unsqueeze(0),dist2.mean(1).unsqueeze(0)])]
#     else:
#         res = [cd_p, cd_t]
#     if calc_f1:
#         f1, _, _ = fscore(dist1, dist2)
#         res.append(f1)
#     if return_raw:
#         res.extend([dist1, dist2, idx1, idx2])
#     return res
#
# def calc_emd(output, gt, eps=0.005, iterations=50):
#     # emd_loss = emd.emdModule()
#     emd_loss = emd()
#     dist, _ = emd_loss(output, gt, eps, iterations)
#     emd_out = torch.sqrt(dist).mean(1)
#     return emd_out
#
# def knn(x, k):
#     inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
#     xx = torch.sum(x ** 2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]
#     return idx
#
# def knn_point(pk, point_input, point_output):
#     m = point_output.size()[1]
#     n = point_input.size()[1]
#
#     inner = -2 * torch.matmul(point_output, point_input.transpose(2, 1).contiguous())
#     xx = torch.sum(point_output ** 2, dim=2, keepdim=True).repeat(1, 1, n)
#     yy = torch.sum(point_input ** 2, dim=2, keepdim=False).unsqueeze(1).repeat(1, m, 1)
#     pairwise_distance = -xx - inner - yy
#     dist, idx = pairwise_distance.topk(k=pk, dim=-1)
#     return dist, idx
#
# def knn_point_all(pk, point_input, point_output):
#     m = point_output.size()[1]
#     n = point_input.size()[1]
#
#     inner = -2 * torch.matmul(point_output, point_input.transpose(2, 1).contiguous())
#     xx = torch.sum(point_output ** 2, dim=2, keepdim=True).repeat(1, 1, n)
#     yy = torch.sum(point_input ** 2, dim=2, keepdim=False).unsqueeze(1).repeat(1, m, 1)
#     pairwise_distance = -xx - inner - yy
#     dist, idx = pairwise_distance.topk(k=pk, dim=-1)
#
#     return dist, idx
KEY_OUTPUT = 'metric_depth'
def extract_key(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction
class GradL1Loss(nn.Module):
    """Gradient loss"""

    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):

        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        grad_gt = grad(target)
        grad_pred = grad(input)



        loss = nn.functional.l1_loss(grad_pred[0], grad_gt[0])
        loss = loss + \
               nn.functional.l1_loss(grad_pred[1], grad_gt[1])
        if not return_interpolated:
            return loss
        return loss, intr_input

def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    mag = diff_x ** 2 + diff_y ** 2
    # angle_ratio
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle

if __name__ == "__main__":
    pc1 = torch.randn([1, 3, 81920]).cuda()

    pc2 =pc1.clone()
    pc2 = torch.randn([1,3,81920]).cuda()

    print(dcd)

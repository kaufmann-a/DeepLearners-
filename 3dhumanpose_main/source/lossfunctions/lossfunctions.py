import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import math


class L1JointRegressionLoss(nn.Module):
    def __init__(self, num_joints, size_average=True, reduce=True, norm=False):
        super(L1JointRegressionLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.num_joints = num_joints
        self.norm = norm

    def weighted_l1_loss(self, input, target, weights, size_average, norm=False):

        if norm:
            input = input / torch.norm(input, 1)
            target = target / torch.norm(target, 1)

        out = torch.abs(input - target)
        out = out * weights
        if size_average:
            num_valid = weights.byte().any(dim=1).float().sum()
            return out.sum() / num_valid
        else:
            return out.sum()

    def _assert_no_grad(self, tensor):
        assert not tensor.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these tensors as not requiring gradients"

    def forward(self, preds, *args):
        gt_joints = args[0]
        gt_joints_vis = args[1]

        pred_jts = preds.reshape((preds.shape[0], self.num_joints * 3))

        self._assert_no_grad(gt_joints)
        self._assert_no_grad(gt_joints_vis)
        return self.weighted_l1_loss(pred_jts, gt_joints, gt_joints_vis, self.size_average, self.norm)
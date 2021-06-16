"""
Class Joint Probabilistic Loss
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"


import torch
from torch.distributions.categorical import Categorical

from source.lossfunctions.l1jointregressionloss import weighted_l1_loss


class JointProbabilisticLoss(torch.nn.Module):
    """Probabilistic joint position selection for which we can derive the
    expected loss w.r.t. to all learnable parameters.

    Inspired by "DSAC - Differentiable RANSAC for Camera Localization" (https://arxiv.org/abs/1611.05705)"""

    def __init__(self, num_samples):
        super().__init__()

        self.num_samples = num_samples

    def forward_one(self, heatmaps, gt_joints, gt_joints_vis):
        J, D, H, W = heatmaps.shape

        heatmaps = heatmaps.reshape(J, D * H * W)
        distribution = Categorical(logits=heatmaps)

        sample_losses = []
        for sample in distribution.sample((self.num_samples,)):
            x_sample = (sample % W)
            x_sample = x_sample / W - 0.5

            y_sample = (sample // W) % H
            y_sample = y_sample / H - 0.5

            z_sample = (sample // (W * H))
            z_sample = z_sample / D - 0.5

            sample_losses.append(weighted_l1_loss(
                torch.stack((x_sample, y_sample, z_sample), dim=1),
                gt_joints,
                gt_joints_vis,
                size_average=False
            ) / -distribution.log_prob(sample))  # REINFORCE (Monte Carlo)

        return torch.stack(sample_losses).mean()

    def forward(self, preds, batch_joints, batch_joints_vis):
        preds, _ = preds
        N, J, _, _, _ = preds.shape

        assert len(batch_joints) == len(batch_joints_vis) == N, \
            f'Batch size mismatch: {N=}, {len(batch_joints)=}, {len(batch_joints_vis)=}'

        batch_losses = []
        for batch_idx in range(N):
            heatmaps = preds[batch_idx]
            gt_joints = batch_joints[batch_idx].reshape(J, 3)
            gt_joints_vis = batch_joints_vis[batch_idx].reshape(J, 3)

            # assert gt_joints_vis.shape == (J, 3), \
            #     f'Invalid shape for batch_joints_vis[{batch_idx}]: expected (J, 3) ' \
            #     f'where {J=} but actual {gt_joints_vis.shape}'

            # assert gt_joints.shape == (J, 3), \
            #     f'Invalid shape for batch_joints[{batch_idx}]: expected (J, 3) ' \
            #     f'where {J=} but actual {gt_joints.shape}'

            batch_losses.append(
                self.forward_one(heatmaps, gt_joints, gt_joints_vis)
            )

        return torch.stack(batch_losses).mean()

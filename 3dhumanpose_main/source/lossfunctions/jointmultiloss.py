import torch

from source.lossfunctions.l2jointheatmaploss import L2JointHeatmapLoss
from source.lossfunctions.l1jointregressionloss import L1JointRegressionLoss


class JointMultiLoss(torch.nn.Module):
    def __init__(self, heatmap_weight, regression_weight):
        super().__init__()

        self.heatmap_loss = L2JointHeatmapLoss()
        self.heatmap_weight = heatmap_weight

        self.regression_loss = L1JointRegressionLoss()
        self.regression_weight = regression_weight

    def forward(self, preds, batch_joints, batch_joints_vis):
        heatmaps, joints = preds
        N, J, D, H, W = heatmaps.shape

        assert joints.shape == (N, J, 3), \
            f"Incompatible shape between heatmaps and joints: expected joints (N, J, 3) " \
            f"where {N=}, {J=} but actual {tuple(joints.shape)}"

        assert batch_joints.shape == (N, J * 3), \
            f"Invalid shape for raw batch_joints: expected (N, 3 * J) " \
            f"where {N=}, {J=} but actual {tuple(batch_joints.shape)}"

        assert batch_joints_vis.shape == (N, J * 3), \
            f"Invalid shape for raw batch_joints_vis: expected (N, 3 * J) " \
            f"where {N=}, {J=} but actual {tuple(batch_joints_vis.shape)}"

        # Reshape batch_joints and batch_joint_vis into coordinate format
        batch_joints = batch_joints.reshape(N, J, 3)
        batch_joints_vis = batch_joints_vis[:, ::3].reshape(N, J)

        # Remove Z coordinates when detecting an MPII batch
        if (batch_joints[:, :, 2] == 0.0).all():
            batch_joints = batch_joints[:, :, :2]

        total_loss = []

        if self.heatmap_weight != 0.0:
            total_loss.append(
                self.heatmap_weight * self.heatmap_loss(heatmaps, batch_joints, batch_joints_vis)
            )

        if self.regression_weight != 0.0:
            total_loss.append(
                self.regression_weight * self.regression_loss(joints, batch_joints, batch_joints_vis)
            )

        return torch.stack(total_loss).sum()

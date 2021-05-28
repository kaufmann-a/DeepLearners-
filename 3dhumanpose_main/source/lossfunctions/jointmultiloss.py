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

        assert batch_joints.shape == (N, J, 3), \
            f"Invalid shape for raw batch_joints: expected (N, J, 3) " \
            f"where {N=}, {J=} but actual {tuple(batch_joints.shape)}"

        assert batch_joints_vis.shape == (N, J), \
            f"Invalid shape for raw batch_joints_vis: expected (N, J) " \
            f"where {N=}, {J=} but actual {tuple(batch_joints_vis.shape)}"

        # Remove Z coordinates when detecting an MPII batch
        # TODO this does not work a batch can be mixed with h36m and mpii! Thus mpii is also a 3d tensor.
        if (batch_joints[:, :, 2] == 0.0).all():
            batch_joints = batch_joints[:, :, :2]

        # TODO: Improve API for multiple losses (and display them seperately in Tensorboard)
        rv = {}
        if self.heatmap_weight != 0.0:
            rv['heatmap_loss'] = self.heatmap_weight * self.heatmap_loss(heatmaps, batch_joints, batch_joints_vis)
        if self.regression_weight != 0.0:
            rv['regression_loss'] = self.regression_weight * self.regression_loss(joints, batch_joints,
                                                                                  batch_joints_vis)
        rv['loss'] = torch.stack(tuple(rv.values())).sum()

        return rv

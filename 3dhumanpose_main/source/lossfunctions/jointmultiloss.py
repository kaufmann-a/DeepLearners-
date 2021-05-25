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

        total_loss = torch.zeros(1, device=heatmaps.device)

        if self.heatmap_weight != 0.0:
            total_loss += self.heatmap_weight * self.heatmap_loss(heatmaps, batch_joints, batch_joints_vis)

        if self.regression_weight != 0.0:
            total_loss += self.regression_weight * self.regression_loss(joints, batch_joints, batch_joints_vis)

        return total_loss

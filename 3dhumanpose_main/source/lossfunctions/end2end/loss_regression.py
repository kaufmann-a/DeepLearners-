import torch


class L1JointRegressionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def regression_loss_2d(pred_joints, gt_joints, gt_joints_vis):
        error = abs(pred_joints[:, :2] - gt_joints) * gt_joints_vis
        return torch.sum(error)

    @staticmethod
    def regression_loss_3d(pred_joints, gt_joints, gt_joints_vis):
        error = abs(pred_joints - gt_joints) * gt_joints_vis
        return torch.sum(error)

    def forward(self, preds, batch_joints, batch_joints_vis):
        N, J, _ = preds.shape

        assert len(batch_joints) == len(batch_joints_vis) == N, \
            f'Batch size mismatch: {N=}, {len(batch_joints)=}, {len(batch_joints_vis)=}'

        batch_losses = []
        for batch_idx in range(N):
            gt_joints = batch_joints[batch_idx]
            gt_joints_vis = batch_joints_vis[batch_idx]

            assert gt_joints_vis.shape == (J,), \
                f'Invalid shape for batch_joints_vis[{batch_idx}]: expected (J,)' \
                f'where {J=} but actual {gt_joints_vis.shape}'

            if gt_joints.shape == (J, 2):
                batch_losses.append(
                    self.regression_loss_2d(preds[batch_idx], gt_joints, gt_joints_vis, self.sigma)
                )
            elif gt_joints.shape == (J, 3):
                batch_losses.append(
                    self.regression_loss_3d(preds[batch_idx], gt_joints, gt_joints_vis, self.sigma)
                )
            else:
                raise AssertionError(
                    f'Invalid shape for batch_joints[{batch_idx}]: expected (J, 2) or (J, 3) '
                    f'where {J=} but actual {gt_joints.shape}'
                )

        return torch.cat(batch_losses).sum()

import math
import torch


def _normal_density_1d(length, device, mu=0., sigma=1.):
    kern = -0.5 * ((torch.arange(length).to(device) - mu) ** 2) / (sigma ** 2)
    return torch.exp(kern) / (sigma * math.sqrt(2 * math.pi))


# Shape must be provided in H, W format
def _normal_density_2d(shape, device, mu_x, mu_y, sigma=1.):
    H, W = shape
    kern_y = _normal_density_1d(H, device, mu_y, sigma)
    kern_x = _normal_density_1d(W, device, mu_x, sigma)
    return kern_y.unsqueeze(0) @ kern_x.unsqueeze(1)


# Shape must be provided in D, H, W format
def _normal_density_3d(shape, device, mu_x, mu_y, mu_z, sigma=1.):
    D, H, W = shape
    kern_z = _normal_density_1d(D, device, mu_z, sigma)
    kern_y = _normal_density_1d(H, device, mu_y, sigma)
    kern_x = _normal_density_1d(W, device, mu_x, sigma)
    return torch.einsum('i, j, k -> ijk', kern_z, kern_y, kern_x)


class L2JointHeatmapLoss(torch.nn.Module):
    def __init__(self, sigma=1.):
        super().__init__()
        self.sigma = sigma

    @staticmethod
    def heatmap_loss_2d(heatmap, gt_joints, gt_joints_vis, sigma=1.):
        J, D, H, W = heatmap.shape

        joint_losses = []
        for j in range(J):
            joint_x, joint_y = gt_joints[j]
            truth_map = _normal_density_2d((H, W), heatmap.device, joint_x, joint_y, sigma)
            error_map = (heatmap[j] - truth_map) ** 2  # Truth map is broadcasted along depth
            error_map = torch.sum(error_map, dim=(1, 2))

            assert error_map.shape == (D,), "2d error map summed incorrectly"

            joint_losses.append(error_map.min())

        joint_losses = torch.stack(joint_losses) * gt_joints_vis
        return torch.sum(joint_losses)

    @staticmethod
    def heatmap_loss_3d(heatmap, gt_joints, gt_joints_vis, sigma=1.):
        J, D, H, W = heatmap.shape

        joint_losses = []
        for j in range(J):
            joint_x, joint_y, joint_z = gt_joints[j]
            truth_map = _normal_density_3d((D, H, W), heatmap.device, joint_x, joint_y, joint_z, sigma)
            error_map = (heatmap[j] - truth_map) ** 2
            joint_losses.append(torch.sum(error_map))

        joint_losses = torch.stack(joint_losses) * gt_joints_vis
        return torch.sum(joint_losses)

    def forward(self, heatmaps, batch_joints, batch_joints_vis):
        N, J, _, _, _ = heatmaps.shape

        assert len(batch_joints) == len(batch_joints_vis) == N, \
            f'Batch size mismatch: {N=}, {len(batch_joints)=}, {len(batch_joints_vis)=}'

        batch_losses = []
        for batch_idx in range(N):
            gt_joints = batch_joints[batch_idx]
            gt_joints_vis = batch_joints_vis[batch_idx]

            assert gt_joints_vis.shape == (J,), \
                f'Invalid shape for batch_joints_vis[{batch_idx}]: expected (J,) ' \
                f'where {J=} but actual {gt_joints_vis.shape}'

            if gt_joints.shape == (J, 2):
                batch_losses.append(
                    self.heatmap_loss_2d(heatmaps[batch_idx], gt_joints, gt_joints_vis, self.sigma)
                )
            elif gt_joints.shape == (J, 3):
                batch_losses.append(
                    self.heatmap_loss_3d(heatmaps[batch_idx], gt_joints, gt_joints_vis, self.sigma)
                )
            else:
                raise AssertionError(
                    f'Invalid shape for batch_joints[{batch_idx}]: expected (J, 2) or (J, 3) '
                    f'where {J=} but actual {gt_joints.shape}'
                )

        return torch.stack(batch_losses).sum()

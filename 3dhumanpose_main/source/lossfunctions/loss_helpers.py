"""
Helper functions for occluder dataset, based on https://github.com/isarandi/synthetic-occlusion

"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

from functools import partial

import numpy as np

from source.configuration import Configuration


def get_label_func():
    # TODO Refactor: Move code into respective loss function and return this function with the loss function
    loss_cfg = Configuration.get('training.loss', optional=False)
    if loss_cfg.loss_function == "JointMultiLoss":
        return generate_integral_joint_location_label
    else:
        # partially apply function: reshape = True
        return partial(generate_joint_location_label, reshape=True)


def generate_integral_joint_location_label(patch_width, patch_height, joints, joints_vis):
    joints[:, 0] = joints[:, 0] / patch_width - 0.5
    joints[:, 1] = joints[:, 1] / patch_height - 0.5
    joints[:, 2] = joints[:, 2] / patch_width

    # check if all dimension have same visibility
    # assert ((joints_vis[:, 0] == joints_vis[:, 1]).all())
    # assert ((joints_vis[:, 1] == joints_vis[:, 2]).all())

    joints_vis = joints_vis[:, 0]  # take only x dimension for visibility

    return joints, joints_vis


def generate_joint_location_label(patch_width, patch_height, joints, joints_vis, reshape=True):
    joints[:, 0] = joints[:, 0] / patch_width - 0.5
    joints[:, 1] = joints[:, 1] / patch_height - 0.5
    joints[:, 2] = joints[:, 2] / patch_width

    if reshape:
        joints = joints.reshape((-1))
        joints_vis = joints_vis.reshape((-1))

    return joints, joints_vis


def get_result_func():
    # TODO Refactor: Move code into respective loss function and return this function with the loss function
    loss_cfg = Configuration.get('training.loss', optional=False)
    if loss_cfg.loss_function == "JointMultiLoss" or \
            loss_cfg.loss_function == 'JointProbabilisticLoss':
        return get_integral_joint_location_result
    return get_joint_regression_result


def get_integral_joint_location_result(patch_width, patch_height, preds):
    heatmap, pred_jts = preds

    pred_jts = pred_jts.reshape((pred_jts.shape[0], -1))

    return get_joint_regression_result(patch_width, patch_height, pred_jts)


def get_joint_regression_result(patch_width, patch_height, preds):
    num_joints = preds.shape[1] // 3

    coords = preds.detach().cpu().numpy()
    coords = coords.astype(float)
    coords = coords.reshape((coords.shape[0], coords.shape[1] // 3, 3))
    # project to original image size
    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * patch_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * patch_height
    coords[:, :, 2] = coords[:, :, 2] * patch_width
    scores = np.ones((coords.shape[0], coords.shape[1], 1), dtype=float)

    # add score to last dimension
    coords = np.concatenate((coords, scores), axis=2)

    return coords

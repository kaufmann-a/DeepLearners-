from functools import partial

import numpy as np

from source.configuration import Configuration
from source.lossfunctions.loss import integral


def generate_joint_location_label(patch_width, patch_height, joints, joints_vis):
    joints[:, 0] = joints[:, 0] / patch_width - 0.5
    joints[:, 1] = joints[:, 1] / patch_height - 0.5
    joints[:, 2] = joints[:, 2] / patch_width

    joints = joints.reshape((-1))
    joints_vis = joints_vis.reshape((-1))
    return joints, joints_vis


def get_integral_joint_location_result(patch_width, patch_height, preds):
    heatmap, pred_jts = preds

    pred_jts = pred_jts.reshape((pred_jts.shape[0], 17 * 3))

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


def get_label_func():
    # TODO Refactor: Maybe move code to factory and return this function with the loss function
    loss_cfg = Configuration.get('training.loss', optional=False)
    if loss_cfg.loss_function == "IntegralJointLocationLoss":
        func = integral.get_label_func(loss_cfg)
        # partially apply function (because it has additionally a config parameter)
        return partial(func, config=loss_cfg)
    return generate_joint_location_label


def get_result_func():
    # TODO Refactor: move code to factory and return this function with the loss function
    loss_cfg = Configuration.get('training.loss', optional=False)
    if loss_cfg.loss_function == "IntegralJointLocationLoss":
        func = integral.get_result_func(loss_cfg)
        # partially apply function (because it has additionally a config parameter)
        return partial(func, config=loss_cfg)
    if loss_cfg.loss_function == "JointMultiLoss":
        return get_integral_joint_location_result
    return get_joint_regression_result

import random

import cv2
import numpy as np
import torch

from source.helpers.transforms import get_2d_rotation
import source.helpers.voc_occluder_helper as voc_occluders
from source.exceptions.configurationerror import VOC_OccluderError

def gen_trans_from_patch_cv(c_x, c_y,
                            src_width, src_height,
                            dst_width, dst_height,
                            scale, rot,
                            inv=False):
    # augment image size with the scale factor
    src_w = src_width.item() * scale
    src_h = src_height.item() * scale

    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y  # np.array([c_x, c_y], dtype=np.float32)

    src_downdir = np.array([0, src_h * 0.5], dtype=np.float32)
    src_rightdir = np.array([src_w * 0.5, 0], dtype=np.float32)

    # augment image with the rotation factor
    rot_rad = np.pi * rot / 180
    src_downdir = get_2d_rotation(src_downdir, rot_rad)
    src_rightdir = get_2d_rotation(src_rightdir, rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


def generate_patch_image_cv(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height,
                            scale_factor, rotation_factor, flip_img):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    if flip_img:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height,
                                    scale=scale_factor, rot=rotation_factor, inv=False)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)

    return img_patch, trans


def convert_cvimg_to_tensor(cvimg):
    # from h,w,c(OpenCV) to c,h,w
    tensor = cvimg.copy()
    tensor = np.transpose(tensor, (2, 0, 1))
    # from BGR(OpenCV) to RGB
    # tensor = tensor[::-1, :, :]
    # from int to float
    tensor = tensor.astype(np.float32)
    return tensor


def trans_coords_from_patch_to_org(coords_in_patch, c_x, c_y, bb_width, bb_height, patch_width, patch_height):
    coords_in_org = coords_in_patch.copy()
    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height,
                                    scale=1.0, rot=.0, inv=True)
    for p in range(coords_in_patch.shape[0]):
        coords_in_org[p, 0:2] = trans_point2d(coords_in_patch[p, 0:2], trans)
    return coords_in_org


def trans_coords_from_patch_to_org_3d(coords_in_patch, c_x, c_y, bb_width, bb_height, patch_width, patch_height
                                      , rect_3d_width, rect_3d_height):
    res_img = trans_coords_from_patch_to_org(coords_in_patch, c_x, c_y, bb_width, bb_height, patch_width, patch_height)
    res_img[:, 2] = coords_in_patch[:, 2] / patch_width * rect_3d_width
    return res_img


def debug_vis_patch(img_patch_cv, joints, joints_vis, flip_pairs, window_name="patch_with_gt"):
    import matplotlib.pyplot as plt
    from source.helpers.vis import cv_draw_joints
    cv_img_patch_show = img_patch_cv.copy()
    cv_draw_joints(cv_img_patch_show, joints, joints_vis, flip_pairs)
    plt.imshow(cv_img_patch_show)
    plt.title(window_name)
    plt.show()
    plt.close()


def get_single_patch_sample(joint_dataset_obj, img_path, center_x, center_y, width, height,
                            patch_width, patch_height,
                            rect_3d_width, rect_3d_height, mean, std,
                            label_func,
                            joint_flip_pairs,
                            apply_augmentations: bool,
                            augmentation_config,
                            joints=None, joints_vis=None,
                            DEBUG=False):
    # 1. load image
    cvimg = cv2.imread(
        img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    if not isinstance(cvimg, np.ndarray):
        raise IOError("Fail to read %s" % img_path)

    img_height, img_width, img_channels = cvimg.shape

    # 2. get augmentation parameters
    if apply_augmentations:
        apply_random_flip, color_scale, rotation, scale = get_augmentation_params(augmentation_config)
    else:
        scale = 1.0
        rotation = 0.0
        apply_random_flip = False
        color_scale = [1.0, 1.0, 1.0]

    if augmentation_config.voc_occluder:
        if np.random.uniform(0.0, 1.0) < augmentation_config.voc_occluder_p:
            try:
                cvimg = voc_occluders.occlude_with_objects(cvimg, joint_dataset_obj.occluders)
            except:
                raise VOC_OccluderError(joint_dataset_obj.occluders)

    # 3. generate image patch
    img_patch_cv, trans = generate_patch_image_cv(cvimg, center_x, center_y, width, height, patch_width, patch_height,
                                                  flip_img=apply_random_flip,
                                                  scale_factor=scale,
                                                  rotation_factor=rotation)

    image = img_patch_cv.copy()
    image = image[:, :, ::-1]

    img_patch_cv = image.copy()
    img_patch = convert_cvimg_to_tensor(image)

    # apply normalization
    for n_c in range(img_channels):
        img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)  # apply color scaling
        if mean is not None and std is not None:
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]

    # 4. (Only for train/val) generate joint ground truth
    if joints is not None and joints_vis is not None:
        if apply_random_flip:
            joints, joints_vis = fliplr_joints(joints, joints_vis, img_width, joint_flip_pairs)

        for n_jt in range(len(joints)):
            joints[n_jt, 0:2] = trans_point2d(joints[n_jt, 0:2], trans)
            joints[n_jt, 2] = joints[n_jt, 2] / (rect_3d_width * scale) * patch_width

        if DEBUG:
            debug_vis_patch(img_patch_cv, joints, joints_vis, joint_flip_pairs)

        # 5. get label of some type according to certain need
        label, label_weight = label_func(patch_width, patch_height, joints, joints_vis)
    else:
        label = np.zeros(1)
        label_weight = np.zeros(1)

    return img_patch, label, label_weight


def get_augmentation_params(augmentation_config):
    """
    Randomly chooses augmentation parameters.

    Args:
        augmentation_config:
            scale_factor: image scale factor
            rotation_factor:
            color_factor:
            random_flip: True = randomly choose to flip the image
                         False = do not flip image

    Returns: random factors: scale, rotation, and color scale
             flipped image: True / false

    """
    rotation_prob = 0.6
    flip_prob = 0.5

    # random scale factor (sf) in range: [1.0 - sf, 1.0 + sf]
    scale = 1.0 + augmentation_config.scale_factor * np.clip(np.random.randn(), a_min=-1.0, a_max=1.0, dtype=np.float32)

    # random rotation factor (rf) in range: rf * [-2.0, 2.0]
    rotation = augmentation_config.rotation_factor * np.clip(np.random.randn(), a_min=-2.0, a_max=2.0, dtype=np.float32) \
        if random.random() <= rotation_prob else 0.0

    # randomly flip image
    apply_random_flip = augmentation_config.random_flip and random.random() <= flip_prob

    # color scaling
    c_up = 1.0 + augmentation_config.color_factor
    c_low = 1.0 - augmentation_config.color_factor
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

    return apply_random_flip, color_scale, rotation, scale


def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)


def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
                            dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    assert flipped.device == tensor.device
    assert flipped.requires_grad == tensor.requires_grad
    return flipped


def fliplr_joints(_joints, _joints_vis, width, matched_parts):
    """
    Flip joint coordinates horizontally.

    Based on: https://github.com/JimmySuen/integral-human-pose/blob/ad3f875bc05538da3471ef81484e23fad3e9c787/common/utility/image_processing_cv.py#L13

    Args:
        _joints: nJoints * dim, dim == 2 [x, y] or dim == 3  [x, y, z]
        _joints_vis:
        width: image width
        matched_parts: list of joint pairs left/right

    Returns: joints, joints_vis

    """
    joints = _joints.copy()
    joints_vis = _joints_vis.copy()

    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints, joints_vis

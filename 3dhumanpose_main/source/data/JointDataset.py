import os

import numpy as np
from torch.utils.data import Dataset

from source.helpers.img_utils import get_single_patch_sample


class JointDataset(Dataset):
    # the unified joints name to idx mapping if we combine datasets
    union_joints = {
        0: 'Hip',
        1: 'RHip',
        2: 'RKnee',
        3: 'RFoot',
        4: 'LHip',
        5: 'LKnee',
        6: 'LFoot',
        7: 'Spine',
        8: 'Thorax',
        9: 'Neck/Nose',
        10: 'Head',
        11: 'LShoulder',
        12: 'LElbow',
        13: 'LWrist',
        14: 'RShoulder',
        15: 'RElbow',
        16: 'RWrist',
    }

    def __init__(self, general_cfg, is_train):
        """
        Based on: https://github.com/yihui-he/epipolar-transformers/blob/4da5cbca762aef6a89d37f889789f772b87d2688/data/datasets/joints_dataset.py

        Args:
            general_cfg: the data collection configuration.
            is_train: True = Is training data set.
        """
        self.dataset_params = getattr(general_cfg, str(self.name) + "_params")
        if is_train:
            image_set = self.dataset_params.train_set
        else:
            image_set = self.dataset_params.val_set

        self.cfg_general = general_cfg
        self.augmentations = self.cfg_general.augmentations

        self.root = os.path.join(general_cfg.folder, self.name)
        self.image_set = image_set
        self.is_train = is_train

        self.num_joints = self.dataset_params.num_joints
        self.num_cams = self.dataset_params.num_cams

        self.patch_width = general_cfg.image_size[0]
        self.patch_height = general_cfg.image_size[1]

        self.rect_3d_width = 2000.
        self.rect_3d_height = 2000.

        self.mean = np.array([123.675, 116.280, 103.530])
        self.std = np.array([58.395, 57.120, 57.375])

        self.label_func = self.get_label_func()

        self.parent_ids = None
        self.flip_pairs = None

        self.db_length = 0

        self.db = []

    def get_joint_mapping(self, actual_joints):
        union_keys = list(self.union_joints.keys())
        union_values = list(self.union_joints.values())

        mapping = {k: '*' for k in union_keys}
        for k, v in actual_joints.items():
            idx = union_values.index(v)
            key = union_keys[idx]
            mapping[key] = k
        return mapping

    def do_joint_mapping(self, mapping):
        for item in self.db:
            joints = item['joints_3d']
            joints_vis = item['joints_3d_vis']

            nr_joints = len(mapping)
            joints_union = np.zeros(shape=(nr_joints, 3))
            joints_union_vis = np.zeros(shape=(nr_joints, 3))

            for i in range(nr_joints):
                if mapping[i] != '*':
                    index = int(mapping[i])
                    joints_union[i] = joints[index]
                    joints_union_vis[i] = joints_vis[index]
            item['joints_3d'] = joints_union
            item['joints_3d_vis'] = joints_union_vis

    def generate_joint_location_label(self, patch_width, patch_height, joints, joints_vis):
        joints[:, 0] = joints[:, 0] / patch_width - 0.5
        joints[:, 1] = joints[:, 1] / patch_height - 0.5
        joints[:, 2] = joints[:, 2] / patch_width

        joints = joints.reshape((-1))
        joints_vis = joints_vis.reshape((-1))
        return joints, joints_vis

    def get_label_func(self):
        return self.generate_joint_location_label

    def __len__(self, ):
        return self.db_length

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_data(self, the_db):
        image_file = os.path.join(self.root, the_db['image'])

        if 'cam' in the_db:
            cam = the_db['cam']
        else:
            cam = None

        if 'joints_3d_vis' in the_db.keys() and 'joints_3d' in the_db.keys():
            joints = the_db['joints_3d'].copy()
            joints_vis = the_db['joints_3d_vis'].copy()
            joints_vis[:, 2] *= self.cfg_general.z_weight  # multiply the z axes of the visibility with z_weight
        else:
            joints = joints_vis = None

        img_patch, label, label_weight = get_single_patch_sample(image_file,
                                                                 the_db['center_x'], the_db['center_y'],
                                                                 the_db['width'], the_db['height'],
                                                                 self.patch_width, self.patch_height,
                                                                 self.rect_3d_width, self.rect_3d_height,
                                                                 self.mean, self.std, self.label_func,
                                                                 joint_flip_pairs=self.flip_pairs,
                                                                 apply_augmentations=self.is_train,
                                                                 augmentation_config=self.augmentations,
                                                                 joints=joints,
                                                                 joints_vis=joints_vis)

        meta = {
            'image': image_file,
            'center_x': the_db['center_x'],
            'center_y': the_db['center_y'],
            'width': the_db['width'],
            'height': the_db['height'],
            'R': cam.R if cam is not None else np.zeros((3, 3), dtype=np.float64),
            'T': cam.T if cam is not None else np.zeros((3, 1), dtype=np.float64),
            'f': cam.f if cam is not None else np.zeros((2, 1), dtype=np.float64),
            'c': cam.c if cam is not None else np.zeros((2, 1), dtype=np.float64),
            'projection_matrix': cam.projection_matrix if cam is not None else np.zeros((3, 4), dtype=np.float64)
        }

        return img_patch.astype(np.float32), label.astype(np.float32), label_weight.astype(np.float32), meta

    def evaluate(self, preds, save_path=None, debug=False):
        raise NotImplementedError

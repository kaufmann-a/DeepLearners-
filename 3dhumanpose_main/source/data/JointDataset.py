import os
import logging
import numpy as np
from torch.utils.data import Dataset


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
        Based on: https://github.com/yihui-he/epipolar-transformers/blob/4da5cbca762aef6a89d37f889789f772b87d2688/data/datasets/joints_dataset.py#L131

        Args:
            general_cfg: the data collection configuration.
            is_train: True = Is training data set.
        """
        dataset_params = getattr(general_cfg, str(self.name) + "_params")
        if is_train:
            image_set = dataset_params.train_set
        else:
            image_set = dataset_params.val_set

        self.cfg_general = general_cfg
        self.root = os.path.join(general_cfg.folder, self.name)
        self.image_set = image_set
        self.is_train = is_train

        self.patch_width = general_cfg.image_size[0]
        self.patch_height = general_cfg.image_size[1]

        self.rect_3d_width = 2000.
        self.rect_3d_height = 2000.

        self.mean = np.array([123.675, 116.280, 103.530])
        self.std = np.array([58.395, 57.120, 57.375])
        self.num_cams = general_cfg.num_cams

        self.label_func = self.get_label_func()

        self.parent_ids = None
        self.db_length = 0

        self.db = []

        self.num_joints = dataset_params.num_joints

        self.actual_joints = {}
        self.u2a_mapping = {}

    def get_joint_mapping(self):
        union_keys = list(self.union_joints.keys())
        union_values = list(self.union_joints.values())

        mapping = {k: '*' for k in union_keys}
        for k, v in self.actual_joints.items():
            idx = union_values.index(v)
            key = union_keys[idx]
            mapping[key] = k
        return mapping

    def do_joint_mapping(self):
        mapping = self.u2a_mapping
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

    def evaluate(self, preds, save_path=None, debug=False):
        raise NotImplementedError

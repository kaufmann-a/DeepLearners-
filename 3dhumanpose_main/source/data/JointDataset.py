"""
Parent Class of Dataset classes, provides methods used for both H36M and MPII
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

import copy
import os

import numpy as np
from torch.utils.data import Dataset

from source.helpers.img_utils import get_single_patch_sample
from source.logcreator.logcreator import Logcreator
import source.helpers.voc_occluder_helper as voc_occluders
import source.exceptions.configurationerror as cfgerror

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

    def __init__(self, general_cfg, label_function, is_train):
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

        self.label_func = label_function

        self.parent_ids = None
        self.flip_pairs = None

        self.parent_ids_super = np.array([0, 0, 1, 2, 0, 4, 5, 0, 8, 8, 9, 8, 11, 12, 8, 14, 15], dtype=np.int)
        self.flip_pairs_super = np.array([[1, 4], [2, 5], [3, 6], [14, 11], [15, 12], [16, 13]], dtype=np.int)

        self.db_length = 0

        self.db = []

        self.occluders = None
        #Prepare voc occluders if used in training
        if self.augmentations.voc_occluder and self.is_train:
            try:
                self.occluders = voc_occluders.load_occluders(self.augmentations.voc_occluder_datapath)
                Logcreator.debug("Occluder data successfully loaded.")
            except:
                Logcreator.error("Occluder could not be initialized, training will be performed without occluder")

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

    def __len__(self, ):
        return self.db_length

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])
        return self.get_data(db_rec, idx)

    def get_data(self, db_rec, idx):
        image_file = os.path.join(self.root, db_rec['image'])

        if 'cam' in db_rec:
            cam = db_rec['cam']
        else:
            cam = None

        if 'joints_3d_vis' in db_rec.keys() and 'joints_3d' in db_rec.keys():
            joints = db_rec['joints_3d'].copy()
            joints_vis = db_rec['joints_3d_vis'].copy()
            joints_vis[:, 2] *= self.cfg_general.z_weight  # multiply the z axes of the visibility with z_weight
        else:
            joints = joints_vis = None

        # apply augmentations only on h36m
        apply_augmentations = self.is_train and (self.name == 'h36m')

        img_patch, label, label_weight = get_single_patch_sample(self, image_file,
                                                                 db_rec['center_x'], db_rec['center_y'],
                                                                 db_rec['width'], db_rec['height'],
                                                                 self.patch_width, self.patch_height,
                                                                 self.rect_3d_width, self.rect_3d_height,
                                                                 self.mean, self.std, self.label_func,
                                                                 joint_flip_pairs=self.flip_pairs_super,
                                                                 apply_augmentations=apply_augmentations,
                                                                 augmentation_config=self.augmentations,
                                                                 joints=joints,
                                                                 joints_vis=joints_vis)

        meta = {
            'image': image_file,
            'name': self.name,
            'idx': idx,
            'center_x': db_rec['center_x'],
            'center_y': db_rec['center_y'],
            'width': db_rec['width'],
            'height': db_rec['height'],
            'R': cam.R if cam is not None else np.zeros((3, 3), dtype=np.float64),
            'T': cam.T if cam is not None else np.zeros((3, 1), dtype=np.float64),
            'f': cam.f if cam is not None else np.zeros((2, 1), dtype=np.float64),
            'c': cam.c if cam is not None else np.zeros((2, 1), dtype=np.float64),
            'projection_matrix': cam.projection_matrix if cam is not None else np.zeros((3, 4), dtype=np.float64)
        }

        return img_patch.astype(np.float32), label.astype(np.float32), label_weight.astype(np.float32), meta

    def evaluate(self, preds, save_path=None, debug=False):
        raise NotImplementedError

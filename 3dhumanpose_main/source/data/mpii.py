import copy
import json
import os

import numpy as np

from source.data.JointDataset import JointDataset
from source.logcreator.logcreator import Logcreator

MPII_PARENT_IDS = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]
MPII_FLIP_PAIRS = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]


class MPIIDataset(JointDataset):
    name = 'mpii'
    actual_joints = {
        0: 'RFoot',
        1: 'RKnee',
        2: 'RHip',
        3: 'LHip',
        4: 'LKnee',
        5: 'LFoot',
        6: 'Hip',
        7: 'Thorax',
        8: 'Neck/Nose',
        9: 'Head',
        10: 'RWrist',
        11: 'RElbow',
        12: 'RShoulder',
        13: 'LShoulder',
        14: 'LElbow',
        15: 'LWrist'
    }

    def __init__(self, general_cfg, is_train):
        super().__init__(general_cfg, is_train)

        self.parent_ids = np.array(MPII_PARENT_IDS, dtype=np.int)
        self.flip_pairs = np.array(MPII_FLIP_PAIRS, dtype=np.int)

        self.pixel_std = 200

        self.db = self._get_db()

        # get joint index to the unified index map
        self.u2a_mapping = super().get_joint_mapping(self.actual_joints)
        if self.image_set != 'test':
            # do mapping of the labels
            super().do_joint_mapping(self.u2a_mapping)

        Logcreator.info('=> load {} samples'.format(self.db_length))

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])
        return self.get_data(db_rec)

    def _get_db(self):
        """
        Loads all images according to the chosen image set.

        Based on https://github.com/yihui-he/epipolar-transformers/blob/4da5cbca762aef6a89d37f889789f772b87d2688/data/datasets/mpii.py
             and https://github.com/JimmySuen/integral-human-pose/blob/master/pytorch_projects/common_pytorch/dataset/mpii.py

        :returns: the ground truth data base
        """
        # create train/val split for MPII
        annotations = self.read_annotation_file()

        # create ground truth database
        gt_db = []
        for idx in range(len(annotations)):
            a = annotations[idx]

            # convert center and scale to 2d arrays representing the x and y axes
            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)  # 2d array with individual scale for x and y

            if c[0] < 1:  # if the center is smaller we skip the image # TODO is this the correct thing to do?
                if False:
                    print(c)
                continue

            # TODO Ask on piazza if the following argumentations are already done to the mpii data that was
            #  supplied by eth

            # we adjust the center and scale slightly to avoid cropping of limbs
            # (this is common practice in multiple git repos)
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]  # adjust the y center coordinate
                s = s * 1.25  # increase the scaling

            # MPII uses matlab format, index is based 1 -> convert it to 0-based index
            c = c - 1

            width = s[0]
            height = s[1]

            width = width * 1.25 * self.pixel_std
            height = height * 1.25 * self.pixel_std

            if width / height >= 1.0 * self.patch_width / self.patch_height:
                width = 1.0 * height * self.patch_width / self.patch_height
            else:
                assert 0, "Error. Invalid patch width and height"

            # joints and vis
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            if self.image_set != 'test':
                jts = np.array(a['joints'])
                jts[:, 0:2] = jts[:, 0:2] - 1
                jts_vis = np.array(a['joints_vis'])
                assert len(jts) == self.num_joints, 'joint num diff: {} vs {}'.format(len(jts), self.num_joints)
                joints_3d[:, 0:2] = jts[:, 0:2]
                joints_3d_vis[:, 0] = jts_vis[:]
                joints_3d_vis[:, 1] = jts_vis[:]

            gt_db.append({
                'image': os.path.join("images", a['image']),
                'center_x': np.asarray([c[0]]),  # convert to array such that it has the same format as h36m
                'center_y': np.asarray([c[1]]),
                'width': np.asarray([width]),
                'height': np.asarray([height]),
                'flip_pairs': self.flip_pairs,
                'parent_ids': self.parent_ids,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        # update the length of the data base
        self.db_length = len(gt_db)

        return gt_db

    def read_annotation_file(self):
        file_name = os.path.join(self.root,
                                 'annot',
                                 self.image_set + '.json')
        with open(file_name, 'rb') as anno_file:
            annotations = json.load(anno_file)
        return annotations

    def evaluate(self, preds, save_path=None, debug=False, writer_dict=None):
        # TODO Do we want to evaluate it or only use it as gradient info.
        pass

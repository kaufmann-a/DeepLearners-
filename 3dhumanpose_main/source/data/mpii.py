import copy
import json
import os

import numpy as np

from source.data.JointDataset import JointDataset
from source.helpers.img_utils import get_single_patch_sample
from source.logcreator.logcreator import Logcreator

MPII_PARENT_IDS = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]
MPII_FLIP_PAIRS = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]

s_mpii_2_hm36_jt = [6, 2, 1, 0, 3, 4, 5, -1, 8, -1, 9, 13, 14, 15, 12, 11, 10, 7]
s_36_jt_num = 18


def from_mpii_to_hm36_single(pose, pose_vis):
    res_jts = np.zeros((s_36_jt_num, 3), dtype=np.float)
    res_vis = np.zeros((s_36_jt_num, 3), dtype=np.float)

    for i in range(0, s_36_jt_num):
        id1 = i
        id2 = s_mpii_2_hm36_jt[i]
        if id2 >= 0:
            res_jts[id1] = pose[id2].copy()
            res_vis[id1] = pose_vis[id2].copy()

    return res_jts.copy(), res_vis.copy()


def from_mpii_to_hm36(db):
    for n_sample in range(0, len(db)):
        res_jts, res_vis = from_mpii_to_hm36_single(db[n_sample]['joints_3d'], db[n_sample]['joints_3d_vis'])
        db[n_sample]['joints_3d'] = res_jts
        db[n_sample]['joints_3d_vis'] = res_vis

class MPIIDataset(JointDataset):
    name = 'mpii'

    def __init__(self, general_cfg, root, image_set, is_train):
        super().__init__(general_cfg, root, image_set, is_train)

        self.parent_ids = np.array(MPII_PARENT_IDS, dtype=np.int)
        self.flip_pairs = np.array(MPII_FLIP_PAIRS, dtype=np.int)

        self.num_joints = 16

        self.num_cams = 1

        self.db = self._get_db()

        Logcreator.info('=> load {} samples'.format(self.db_length))

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])
        return self.get_data(db_rec)

    def get_data(self, the_db):
        image_file = os.path.join(self.root, "images", the_db['image'])

        if 'joints_3d_vis' in the_db.keys() and 'joints_3d' in the_db.keys():
            joints = the_db['joints_3d'].copy()
            joints_vis = the_db['joints_3d_vis'].copy()
            joints_vis[:, 2] *= self.cfg_general.z_weight  # multiply the z axes of the visibility with z_weight
        else:
            joints = joints_vis = None

        # TODO add image transformations similar to get_single_patch_sample
        #  maybe use this as inspiration: https://github.com/yihui-he/epipolar-transformers/tree/4da5cbca762aef6a89d37f889789f772b87d2688/data/datasets
        width = the_db['width']
        height = the_db['height']

        img_patch, label, label_weight = get_single_patch_sample(image_file,
                                                                 the_db['center_x'], the_db['center_y'],
                                                                 width, height,
                                                                 self.patch_width, self.patch_height,
                                                                 self.rect_3d_width, self.rect_3d_height,
                                                                 self.mean, self.std, self.label_func,
                                                                 joints=joints,
                                                                 joints_vis=joints_vis, )
        # DEBUG=self.cfg_general.DEBUG.DEBUG)  # TODO add debug parameter

        meta = {
            'image': image_file,
            'center_x': the_db['center_x'],
            'center_y': the_db['center_y'],
            'width': width,
            'height': height,
            'R': np.zeros((3, 3), dtype=np.float64),
            'T': np.zeros((3, 1), dtype=np.float64),
            'f': np.zeros((2, 1), dtype=np.float64),
            'c': np.zeros((2, 1), dtype=np.float64),
            'projection_matrix': np.zeros((3, 4), dtype=np.float64)
        }

        return img_patch.astype(np.float32), label.astype(np.float32), label_weight.astype(np.float32), meta

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

            # TODO Ask on piazza if the following argumentations are already done to the mpii data that was
            #  supplied by eth

            # we adjust the center and scale slightly to avoid cropping of limbs
            # (this is common practice in multiple git repos)
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]  # adjust the y center coordinate
                s = s * 1.25  # increase the scaling

            # MPII uses matlab format, index is based 1 -> convert it to 0-based index
            assert c[0] >= 1  # checks for matlab format?
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
                'image': a['image'],
                'center_x': c[0],
                'center_y': c[1],
                'width': width,
                'height': height,
                'flip_pairs': self.flip_pairs,
                'parent_ids': self.parent_ids,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        # update the length of the data base
        self.db_length = len(gt_db)

        from_mpii_to_hm36(gt_db)

        return gt_db

    def read_annotation_file(self):
        file_name = os.path.join(self.root,
                                 'annot',
                                 self.image_set + '.json')
        with open(file_name, 'rb') as anno_file:
            annotations = json.load(anno_file)
        return annotations

    def evaluate(self, preds, save_path=None, debug=False):
        # TODO Do we want to evaluate it or only use it as gradient info.
        pass

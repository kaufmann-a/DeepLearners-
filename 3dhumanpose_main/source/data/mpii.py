import copy
import json
import os

import numpy as np

from source.data.JointDataset import JointDataset
from source.helpers.img_utils import get_single_patch_sample
from source.logcreator.logcreator import Logcreator

MPII_PARENT_IDS = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]
MPII_FLIP_PAIRS = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]


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

        center_x = the_db['center'][0]
        center_y = the_db['center'][1]

        # TODO add image transformations similar to get_single_patch_sample
        #  maybe use this as inspiration: https://github.com/yihui-he/epipolar-transformers/tree/4da5cbca762aef6a89d37f889789f772b87d2688/data/datasets
        width = None
        height = None
        img_patch, label, label_weight = get_single_patch_sample(image_file,
                                                                 center_x, center_y,
                                                                 width, height,
                                                                 self.patch_width, self.patch_height,
                                                                 self.rect_3d_width, self.rect_3d_height,
                                                                 self.mean, self.std, self.label_func,
                                                                 joints=the_db['joints'],
                                                                 joints_vis=the_db['joints_vis'], )
        # DEBUG=self.cfg_general.DEBUG.DEBUG)  # TODO add debug parameter

        meta = {
            'image': image_file,
            'center_x': center_x,
            'center_y': center_y,
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

        :returns: the ground truth data base
        """
        # create train/val split for MPII
        annotations = self.read_annotation_file()

        # create ground truth database
        gt_db = []
        for idx in range(len(annotations)):
            a = annotations[idx]
            # TODO Maybe we want to do some preprocessing here? -> otherwise remove loop
            gt_db.append(a)

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

    def evaluate(self, preds, save_path=None, debug=False):
        # TODO Do we want to evaluate it or only use it as gradient info.
        pass

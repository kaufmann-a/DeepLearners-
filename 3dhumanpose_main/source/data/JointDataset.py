import os
import logging
import numpy as np
from torch.utils.data import Dataset
from source.configuration import Configuration

H36M_NAMES = ['']*17
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[4]  = 'LHip'
H36M_NAMES[5]  = 'LKnee'
H36M_NAMES[6]  = 'LFoot'
H36M_NAMES[7] = 'Spine'
H36M_NAMES[8] = 'Thorax'
H36M_NAMES[9] = 'Neck/Nose'
H36M_NAMES[10] = 'Head'
H36M_NAMES[11] = 'LShoulder'
H36M_NAMES[12] = 'LElbow'
H36M_NAMES[13] = 'LWrist'
H36M_NAMES[14] = 'RShoulder'
H36M_NAMES[15] = 'RElbow'
H36M_NAMES[16] = 'RWrist'

MPII_NAMES = ['']*16
MPII_NAMES[0]  = 'RFoot'
MPII_NAMES[1]  = 'RKnee'
MPII_NAMES[2]  = 'RHip'
MPII_NAMES[3]  = 'LHip'
MPII_NAMES[4]  = 'LKnee'
MPII_NAMES[5]  = 'LFoot'
MPII_NAMES[6]  = 'Hip'
MPII_NAMES[7]  = 'Thorax'
MPII_NAMES[8]  = 'Neck/Nose'
MPII_NAMES[9]  = 'Head'
MPII_NAMES[10] = 'RWrist'
MPII_NAMES[11] = 'RElbow'
MPII_NAMES[12] = 'RShoulder'
MPII_NAMES[13] = 'LShoulder'
MPII_NAMES[14] = 'LElbow'
MPII_NAMES[15] = 'LWrist'

logger = logging.getLogger(__name__)

H36M_TO_MPII_PERM = np.array([H36M_NAMES.index(h) for h in MPII_NAMES if h != '' and h in H36M_NAMES])


class JointDataset(Dataset):
    def __init__(self, general_cfg, is_train):
        dataset_params = getattr(general_cfg, str(self.name) + "_params")
        if is_train:
            image_set = dataset_params.train_set
        else:
            image_set = dataset_params.val_set

        """

        Args:
            general_cfg:
            root: The root directory of the data set containing a folder 'annot' (annotations) and a folder 'images'.
            image_set: The name of data set annotation file without the file extension:
                       e.g. train/val/trainval/test
            is_train: True = Is training data set.
        """

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

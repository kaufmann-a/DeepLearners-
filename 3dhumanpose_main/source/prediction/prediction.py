#!/usr/bin/env python3
# coding: utf8

"""
Class for prediction of a set of images
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

import torch
from torch.utils.data import DataLoader

from source.configuration import Configuration
from source.data.datasetfactory import DataSetFactory


class Prediction(object):

    def __init__(self, engine, device):
        """

        :param engine:
        :param device:
       """
        self.engine = engine
        self.device = device

    @staticmethod
    def get_dataset_config():
        """
        TODO Very ugly; Probably better to ad a parameter for the test set name or ...?

        Forces dataset to be "h36m" and the attribute h36m_params.val_set to be "test".

        Returns: data_collection config

        """
        dataset_config = Configuration.get('data_collection')
        dataset_config.dataset = ['h36m']
        dataset_config.h36m_params.val_set = "test"

        return dataset_config

    def predict(self):
        cfg = self.get_dataset_config()

        test_dataset = DataSetFactory.load(cfg, is_train=False)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Configuration.get("training.general.batch_size"),  # * len(gpus)
            shuffle=False,
            num_workers=Configuration.get("training.general.num_workers"),
            pin_memory=True
        )

        # run validate and evaluate with test set
        val_loss, preds_in_patch_with_score = self.engine.validate(test_loader, epoch=0, only_prediction=True)
        acc = self.engine.evaluate(0, preds_in_patch_with_score, test_loader, Configuration.output_directory,
                                   debug=False)

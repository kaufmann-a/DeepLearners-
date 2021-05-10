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

    def predict(self):
        test_dataset = DataSetFactory.load(Configuration.get('data_collection'),
                                           Configuration.get_path('data_collection.folder'),
                                           image_set="test",
                                           is_train=False)

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
                                   debug=False,
                                   writer_dict=None)

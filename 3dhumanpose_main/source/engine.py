#!/usr/bin/env python3
# coding: utf8

"""
Model of the road segmentatjion neuronal network learning object.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

import os

import numpy as np
import torch
#import torchmetrics as torchmetrics
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
# from tqdm import tqdm

from source.configuration import Configuration

from source.logcreator.logcreator import Logcreator
from source.lossfunctions.lossfunctionfactory import LossFunctionFactory


from source.models.modelfactory import ModelFactory
from source.optimizers.optimizerfactory import OptimizerFactory
from source.scheduler.lr_schedulerfactory import LRSchedulerFactory
from source.data.datasetfactory import DataSetFactory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Engine:

    def __init__(self):
        self.model = ModelFactory.build().to(DEVICE)
        self.optimizer = OptimizerFactory.build(self.model)
        self.lr_scheduler = LRSchedulerFactory.build(self.optimizer)
        self.loss_function = LossFunctionFactory.build(self.model).to(DEVICE)
        # self.scaler = torch.cuda.amp.GradScaler()  # I assumed we always use gradscaler, thus no factory for this

        # initialize tensorboard logger
        # Configuration.tensorboard_folder = os.path.join(Configuration.output_directory, "tensorboard")
        # if not os.path.exists(Configuration.tensorboard_folder):
        #     os.makedirs(Configuration.tensorboard_folder)
        # self.writer = SummaryWriter(log_dir=Configuration.tensorboard_folder)

        # Print model summary
        #Logcreator.info(summary(self.model, input_size=input_size, device=DEVICE))

        # Logcreator.debug("Model '%s' initialized with %d parameters." %
        #                  (Configuration.get('training.model.name'),
        #                   sum(p.numel() for p in self.model.parameters() if p.requires_grad)))


    def train(self, epoch_nr=0):

        training_data = DataSetFactory.load(Configuration.get('training.general'), Configuration.get_path('data_collection.folder'),
                                                             Configuration.get('data_collection.train_set'), True)

        training_data = DataSetFactory.load(Configuration.get('training.general'),
                                            Configuration.get_path('data_collection.folder'),
                                            Configuration.get('data_collection.val_set'), False)

        #Todo: Everything from here







    def save_checkpoint(self, epoch, tl, ta, vl, va):
        Configuration.weights_save_folder = os.path.join(Configuration.output_directory, "weights_checkpoint")
        if not os.path.exists(Configuration.weights_save_folder):
            os.makedirs(Configuration.weights_save_folder)
        file_name = str(epoch) + "_checkpoint.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'train_loss': tl,
            'train_accuracy': ta,
            'val_loss': vl,
            'val_accuracy': va,
        }, os.path.join(Configuration.weights_save_folder, file_name))


    def load_checkpints(self, path=None):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict']).to(DEVICE)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        train_accuracy = checkpoint['train_accuracy']
        val_loss = checkpoint['val_loss']
        val_accuracy = checkpoint['val_accuracy']

        return epoch, train_loss, train_accuracy, val_loss, val_accuracy

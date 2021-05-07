#!/usr/bin/env python3
# coding: utf8

"""
Builds a learning rate scheduler from configuration.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

import torch.optim as optim

from source.configuration import Configuration


class LRSchedulerFactory(object):
    optimizer = None

    @staticmethod
    def build(optimizer):
        LRSchedulerFactory.optimizer = optimizer
        scheduler_cfg = Configuration.get('training.lr_scheduler')
        return getattr(LRSchedulerFactory, scheduler_cfg.name)(LRSchedulerFactory, scheduler_cfg)

    def stepLR(self, cfg):
        return optim.lr_scheduler.StepLR(self.optimizer,
                                         step_size=cfg.stepLR.step_size,
                                         gamma=cfg.stepLR.gamma)

    def multiStepLR(self, cfg):
        return optim.lr_scheduler.MultiStepLR(self.optimizer,
                                              milestones=cfg.multiStepLR.milestones,
                                              gamma=cfg.multiStepLR.gamma)

    @staticmethod
    def get_members():
        return {
            'stepLR': LRSchedulerFactory.stepLR,
            'multiStepLR': LRSchedulerFactory.multiStepLR
        }

#!/usr/bin/env python3
# coding: utf8

"""
Builds an optimizer from configuration.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

import torch.optim as optim
from source.configuration import Configuration


class OptimizerFactory(object):
    model = False

    @staticmethod
    def build(model):
        OptimizerFactory.model = model
        optimizer_cfg = Configuration.get('training.optimizer')
        return getattr(OptimizerFactory, optimizer_cfg.name)(OptimizerFactory, optimizer_cfg)

    def adam(self, cfg):
        return optim.Adam(self.model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

    def sdg(self, cfg):
        return optim.SDG(self.model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.wd, nesterov=cfg.nesterov)

    @staticmethod
    def get_members():
        return {
            'adam': OptimizerFactory.adam,
            'sdg': OptimizerFactory.sdg
        }

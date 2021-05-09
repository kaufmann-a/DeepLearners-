#!/usr/bin/env python3
# coding: utf8

"""
Builds a torch loss function from configuration.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

from source.configuration import Configuration
from source.lossfunctions.loss import integral
from source.lossfunctions.lossfunctions import L1JointRegressionLoss


class LossFunctionFactory(object):
    model = False

    @staticmethod
    def build(model):
        LossFunctionFactory.model = model
        loss_cfg = Configuration.get('training.loss', optional=False)
        return getattr(LossFunctionFactory, loss_cfg.loss_function)(LossFunctionFactory, loss_cfg)

    @staticmethod
    def L1JointRegressionLoss(self, cfg):
        return L1JointRegressionLoss(num_joints=Configuration.get('training.model.num_joints'), norm=cfg.norm)

    @staticmethod
    def IntegralJointLocationLoss(self, cfg):
        return integral.get_loss_func(cfg)

    @staticmethod
    def get_members():
        return {
            'L1JointRegressionLoss': LossFunctionFactory.L1JointRegressionLoss,
            'IntegralJointLocationLoss': LossFunctionFactory.IntegralJointLocationLoss,
        }

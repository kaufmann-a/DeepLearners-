#!/usr/bin/env python3
# coding: utf8

"""
Builds a torch loss function from configuration.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

from source.configuration import Configuration
from source.lossfunctions.jointmultiloss import JointMultiLoss
from source.lossfunctions.l1jointregressionloss import L1JointRegressionLoss
from source.lossfunctions.l1jointregressionloss import L1JointRegressionLoss_eth_code
from source.lossfunctions.l2jointheatmaploss import L2JointHeatmapLoss


class LossFunctionFactory(object):
    model = False

    @staticmethod
    def build(model):
        LossFunctionFactory.model = model
        loss_cfg = Configuration.get('training.loss', optional=False)
        return getattr(LossFunctionFactory, loss_cfg.loss_function)(LossFunctionFactory, loss_cfg)

    @staticmethod
    def L1JointRegressionLoss_eth_code(self, cfg):
        return L1JointRegressionLoss_eth_code(num_joints=Configuration.get('training.model.num_joints'), norm=cfg.norm)

    @staticmethod
    def L1JointRegressionLoss(self, cfg):
        return L1JointRegressionLoss()

    @staticmethod
    def L2JointHeatMapLoss(self, cfg):
        return L2JointHeatmapLoss()

    @staticmethod
    def JointMultiLoss(self, cfg):
        return JointMultiLoss(cfg.loss_weight_heatmap, cfg.loss_weight_regression)

    @staticmethod
    def get_members():
        return {
            'L1JointRegressionLoss_eth_code': LossFunctionFactory.L1JointRegressionLoss_eth_code,
            'L1JointRegressionLoss': LossFunctionFactory.L1JointRegressionLoss,
            'L2JointHeatMapLoss': LossFunctionFactory.L2JointHeatMapLoss,
            'JointMultiLoss': LossFunctionFactory.JointMultiLoss
        }

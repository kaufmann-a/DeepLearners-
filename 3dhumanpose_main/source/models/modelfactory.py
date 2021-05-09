#!/usr/bin/env python3
# coding: utf8

"""
Builds a torch model from configuration.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

import glob
# Load all models
from os.path import dirname, basename, isfile

from source.configuration import Configuration
from source.exceptions.configurationerror import ConfigurationError
from source.models.basemodel import BaseModel
from source.logcreator.logcreator import Logcreator
from source.models.blocks import resnet_pose, resnet_direct_regression

modules = glob.glob(dirname(__file__) + "/*.py")
for module in [basename(f)[:-3] for f in modules if
               isfile(f) and not f.endswith('__init__.py') and not f == "modelfactory"]:
    __import__("source.models." + module)


class ModelFactory(object):

    @staticmethod
    def build():
        model_config = Configuration.get('training.model', optional=False)

        all_models = BaseModel.__subclasses__()
        if model_config.name:
            model = [m(model_config) for m in all_models if m.name.lower() == model_config.name.lower()]
            if model and len(model) > 0:
                return model[0]
        # raise ConfigurationError('training.model.name')

        # TODO Maybe refactor to only use one method to get models
        return getattr(ModelFactory, model_config.name)(ModelFactory, model_config)

    @staticmethod
    def ResPoseNet_DeconvHead(self, cfg):
        default_config = resnet_pose.get_default_network_config()  # TODO use parameters from configuration file
        default_config.depth_dim = 64
        net = resnet_pose.get_pose_net(default_config, cfg.num_joints)
        # initialize weights of backbone network
        resnet_pose.init_pose_net(net, default_config)  # TODO add configuration parameter to set loading on/off
        return net

    @staticmethod
    def ResPoseNet_Regression(self, cfg):
        default_config = resnet_direct_regression.get_default_network_config()  # TODO use parameters from configuration file
        default_config.fea_map_size = 8
        net = resnet_direct_regression.get_pose_net(default_config, cfg.num_joints)
        # initialize weights of backbone network
        resnet_direct_regression.init_pose_net(net,
                                               default_config)  # TODO add configuration parameter to set loading on/off
        return net

    @staticmethod
    def get_members():
        return {
            'ResPoseNet_DeconvHead': ModelFactory.ResPoseNet_DeconvHead,
            'ResPoseNet_Regression': ModelFactory.ResPoseNet_Regression,
        }

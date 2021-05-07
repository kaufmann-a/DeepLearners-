#!/usr/bin/env python3
# coding: utf8

"""
Class for prediction of a set of images
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

import math
import os

import torch
from PIL import Image, ImageChops
from matplotlib import pyplot
from torch.utils.data import DataLoader


from source.configuration import Configuration




class Prediction(object):

    def __init__(self, engine, images, device, threshold, use_original_image_size):
        """

        :param engine:
        :param images:
        :param device:
        :param threshold:
        :param use_original_image_size: False = patch image together from small subpatches of same size as in training
        """

        self.device = device
        self.model = engine.model
        self.model.to(device)
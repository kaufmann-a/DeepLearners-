#!/usr/bin/env python3
# coding: utf8

"""
Base model which all models inherit from
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

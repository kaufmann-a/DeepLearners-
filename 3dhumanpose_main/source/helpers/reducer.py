#!/usr/bin/env python3
# coding: utf8

"""
Helps reducing and handle objects.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"


import functools


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def lflatter(input, times=1):
    if len(input.shape) == 2:
        output = input.flatten()
    else:
        output = input.reshape(-1, *input.shape[2:])
    if times > 1:
        return lflatter(output, times - 1)
    return output


def rflatter(input, times=1):
    if len(input.shape) == 2:
        output = input.flatten()
    else:
        output = input.reshape(*input.shape[:-2], -1)
    if times > 1:
        return rflatter(output, times - 1)
    return output
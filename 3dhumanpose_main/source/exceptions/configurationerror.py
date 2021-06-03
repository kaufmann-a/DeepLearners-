#!/usr/bin/env python3
# coding: utf8

"""
Thrown if configuration is missing or invalid.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

class ConfigurationError(Exception):

    def __init__(self, key='', value=False):
        if key and value:
            super().__init__("Invalid configuration for key '%s' with value '%s'." % (key, value))
        elif key:
            super().__init__("Invalid or missing configuration for key '%s'." % key)
        else:
            super().__init__("Configuration missing or not initialized.")


class VOC_OccluderError(Exception):
    def __init__(self, occluders):
        if occluders is None:
            super().__init__("Occlusion didn't conntain any images, make sure VOC dataset is available")
        else:
            super().__init__("Some error in voc occluder occured")

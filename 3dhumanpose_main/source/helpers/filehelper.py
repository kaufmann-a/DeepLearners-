
"""
Helps handling files and folders.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"


import glob
import os


def build_abspath(path, working_directory=''):
    if not os.path.isabs(path):
        if not working_directory:
            working_directory = os.getcwd()
        path = os.path.join(working_directory, path)
    return path


def get_latest(folder, pattern):
    """
    Gets the latest file in a folder matching the naming pattern.
    """
    files = glob.glob(os.path.join(folder, pattern))
    return max(files, key=os.path.getctime)
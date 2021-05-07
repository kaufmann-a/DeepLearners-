import argparse
"""
Handles arguments provided in comand line
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

from source.logcreator.logcreator import Logcreator


def get_args():
    """
    Returns list of args
    """
    return args

def boolean_string(s):
    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def parse_args(parser):
    """
    Read and parse args from comandline and store in args
    """
    if parser:
        global args
        args = parser.parse_args()
    else:
        raise EnvironmentError(
            Logcreator.info("Parsing of comand line parameters failed")
        )
    return args
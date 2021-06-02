"""
Initializes config, loads configfiles, adds folder for run-logs
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

import commentjson
import os
import inspect

import datetime
from shutil import copy
from Lib.types import SimpleNamespace
from source.helpers import converter
from source.helpers import dictionary
from source.helpers import environmentvariables
from source.helpers import reducer
from source.helpers import filehelper


class Configuration(object):
    output_directory = ''

    @staticmethod
    def initialize(configuration_file, working_directory=None, create_output_train=False, create_output_inf=False,
                   disable_merge=False):
        global configuration
        if not working_directory:
            working_directory = os.getcwd()
        environmentvariables.initialize(extend=True)

        if not configuration_file:
            raise EnvironmentError(
                "Configuration file " + configuration_file + "not found - aborting"
            )
        configuration_dict = Configuration.load_config(configuration_file)
        configuration = dictionary.to_namespace(configuration_dict)

        if inspect.getouterframes(inspect.currentframe(), 2)[1].filename.__contains__('train'):
            Configuration.output_directory = os.path.join(working_directory,
                                                          Configuration.get(
                                                              'environment.output_path', optional=False),
                                                          datetime.datetime.now().strftime(
                                                              '%Y%m%d-%H%M%S') + "-" + os.path.basename(
                                                              configuration_file.replace(".jsonc", "")))
        elif inspect.getouterframes(inspect.currentframe(), 2)[1].filename.__contains__('inference'):
            Configuration.output_directory = os.path.join(working_directory,
                                                          'prediction-' + datetime.datetime.now().strftime(
                                                              '%Y%m%d-%H%M%S'))
        else:
            Configuration.output_directory = working_directory
        if not os.path.exists(Configuration.output_directory):
            os.makedirs(Configuration.output_directory)
            copy(configuration_file, Configuration.output_directory)

    @staticmethod
    def load_config(configuration_path):
        if not os.path.exists(configuration_path):
            raise EnvironmentError(
                "Configuration file " + configuration_path + " not found - aborting.")
        with open(Configuration.build_path(configuration_path, Configuration.output_directory, create=False), 'r') as f:
            config = commentjson.load(f, object_hook=lambda d: {
                k: converter.try_eval(d[k]) for k in d})
            return config

    @staticmethod
    def get(key='', optional=False, default=None):
        global configuration
        if not configuration:
            raise EnvironmentError(
                "No configuration existing"
            )
        if key:
            try:
                value = reducer.rgetattr(configuration, key)
                if value is None:
                    if optional:
                        return default
                    raise EnvironmentError(
                        "Invalid key"
                    )
                if isinstance(value, str):
                    if 'getenv' in value:
                        return eval('os.' + value)
                if isinstance(value, SimpleNamespace):
                    getenv_recursive(value)
                    return value
                return value
            except:
                if not optional:
                    raise EnvironmentError(
                        "Invalid key or wrong key specification"
                    )
        return default

    @staticmethod
    def get_path(key='', create=True, optional=False):
        path = Configuration.get(key, optional)
        return Configuration.build_path(path)

    @staticmethod
    def build_path(path, workingdir='', create=True):
        """
        Generates an absolute path if a relative is passed.
        """
        path = filehelper.build_abspath(path, workingdir)
        if create and not os.path.exists(path):
            os.makedirs(path)
        return path


def getenv_recursive(value):
    for element in value.__dict__:
        if isinstance(value.__dict__[element], SimpleNamespace):
            getenv_recursive(value.__dict__[element])
        else:
            if isinstance(value.__dict__[element], str):
                if 'getenv' in value.__dict__[element]:
                    value.__dict__[element] = eval('os.' + value.__dict__[element])


#!/usr/bin/env python3
# coding: utf8

"""
Main class for inference
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

import argparse
import glob
import os
import sys
import time

import torch

from source.configuration import Configuration
from source.engine import Engine
from source.helpers import filehelper
from source.logcreator.logcreator import Logcreator
from source.prediction.prediction import Prediction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def process_args(args):
    if args.run_folder:
        args.workingdir = filehelper.build_abspath(args.run_folder,
                                                   os.getcwd())  # In case relative path was passed in args

        # Load ConfigFile
        args.configuration = glob.glob(
            os.path.join(args.workingdir, '*.jsonc'))  # Assuming only the config file is of type jsonc
        if len(args.configuration) != 1:
            print("More than one config file found or no config file found at all - abort!")
            sys.exit()
        else:
            args.configuration = args.configuration[0]

        # Get weights
        try:
            args.weights = filehelper.get_latest(os.path.join(args.workingdir, 'weights_checkpoint'), '*.pth')
        except ValueError:
            print("No weights found, make sure you use a training with stored weights")
            sys.exit()

        # Init config and logger
        Configuration.initialize(args.configuration, args.workingdir, create_output_train=False, create_output_inf=True)
        Logcreator.initialize(False)  # Don't write a logfile
    else:
        print("No folder of a previous training run provided, add one in arguments")


if __name__ == "__main__":
    global config

    parser = argparse.ArgumentParser(
        description="Does prediction on predefined set of images"
    )
    parser.add_argument('--run_folder', default='',
                        type=str, help="Input here the folder path of the training run you want to use for inference")

    args = parser.parse_args()
    start = time.time()

    process_args(args)

    Logcreator.h1("This is a prediction run with test-images")

    # Init engine
    engine = Engine()
    engine.load_checkpoints(args.weights)

    # Run predictions
    predictor = Prediction(engine=engine, device=DEVICE)
    predictor.predict()

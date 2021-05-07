#!/usr/bin/env python3
# coding: utf8

"""
Logs prettified information to console and output file.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

import os
import time as sys_time

from source.configuration import Configuration
from source.helpers import converter
from source.logcreator.colors import Colors


class Logcreator(object):

    timers = {}
    log_file = ''

    @staticmethod
    def initialize(write_file=True):
        Logcreator.timers = {}
        if write_file:
            Logcreator.log_file = os.path.join(
                Configuration.output_directory, Configuration.get('environment.log_file', optional=False))

    @staticmethod
    def format(iterable):
        return " ".join(str(i) for i in iterable)

    @staticmethod
    def h1(*args):
        print(Colors.DK_WHITE + Logcreator.format(args) + Colors.END)
        Logcreator._write_log("", Logcreator.format(args))

    @staticmethod
    def wait(*args):
        input(Colors.CYAN + Logcreator.format(args) + Colors.END)
        Logcreator._write_log("Waiting", Logcreator.format(args))

    @staticmethod
    def info(*args):
        print(Colors.DIM + "\t", Logcreator.format(args), Colors.END)
        Logcreator._write_log("Info", Logcreator.format(args))

    @staticmethod
    def debug(*args):
        print(Colors.GREEN + "\t", Logcreator.format(args), Colors.END)
        Logcreator._write_log("Debug", Logcreator.format(args))

    @staticmethod
    def warn(*args):
        print(Colors.DK_MAGENTA + "WARN:\t" + Colors.END +
              Colors.MAGENTA, Logcreator.format(args), Colors.END)
        Logcreator._write_log("Warning", Logcreator.format(args))

    @staticmethod
    def error(*args):
        print(Colors.DK_RED + Colors.BLINK + "ERROR:\t" +
              Colors.END + Colors.RED, Logcreator.format(args), Colors.END)
        Logcreator._write_log("Error", Logcreator.format(args))

    @staticmethod
    def time(key):
        Logcreator.timers[key] = sys_time.time()

    @staticmethod
    def time_end(key):
        if key in Logcreator.timers:
            t = sys_time.time() - Logcreator.timers[key]
            print("\t" + str(t) + Colors.DIM + " s \t" + key + Colors.END)
            del Logcreator.timers[key]

    @staticmethod
    def notify(*args):
        # Play bell
        print('\a')

    @staticmethod
    def _write_log(title, message):
        try:
            if not Logcreator.log_file:
                return
            if title:
                content = "%s\t%s: %s" % (
                    converter.get_timestamp(), title, message)
            else:
                content = "%s\t%s" % (
                    converter.get_timestamp(), message)
            with open(Logcreator.log_file, "a", newline='\n', encoding='utf8') as file:
                file.write(content + '\n')
        except:
            pass

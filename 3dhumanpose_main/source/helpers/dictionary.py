#!/usr/bin/env python3
# coding: utf8

"""
Dictinoary helpers
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

from functools import singledispatch
from types import SimpleNamespace


# Deep merges dictionaries, by modifying destination in place. Source: https://stackoverflow.com/a/20666342/496950
def merge(source, destination):
    """
    run me with nosetests --with-doctest file.py

    >>> a = { 'first' : { 'all_rows' : { 'pass' : 'dog', 'number' : '1' } } }
    >>> b = { 'first' : { 'all_rows' : { 'fail' : 'cat', 'number' : '5' } } }
    >>> merge(b, a) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            merge(value, node)
        else:
            destination[key] = value

    return destination


@singledispatch
def to_namespace(ob):
    return ob


@to_namespace.register(dict)
def _namespace_dict(ob):
    return SimpleNamespace(**{k: to_namespace(v) for k, v in ob.items()})


@to_namespace.register(list)
def _namespace_list(ob):
    return [to_namespace(v) for v in ob]

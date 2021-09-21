import functools


import numpy as np


def cached_property(func):
    """Cache a property within a class.
    Requires the Class to have a cache dict called ``_cache``.
    Example
    -------
    How to add a cache for a variable to a class by using the `@cached_property`
    decorator::
        class A(object):
            def__init__(self):
                self._cache = dict()
            @cached_property
            def size(self):
                # This code gets run only if the lookup of keyname fails
                # After this code has been ran once, the result is stored in
                # _cache with the key: 'keyname'
                return size
    .. note::
        Adapted from MDAnalysis. This code is GPL licensed.
    """

    key = func.__name__

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return self._cache[key]
        except KeyError:
            self._cache[key] = ret = func(self, *args, **kwargs)
            return ret
    return property(wrapper)


def get_index_dict(subarray, superarray):
    """
    Get dictionary of indices such that
    ``superarray[indices.keys()] == subarray[indices.values()]``.
    ``subarray` and ``superarray`` *must* both be unique.
    """
    mapping = {}
    for i, j in enumerate(subarray):
        found = np.where(superarray == j)[0]
        if not len(found):
            continue
        mapping[found[0]] = i
    return mapping


def axis_to_index(x):
    return {"x": 0, "y": 1, "z": 2}[x]

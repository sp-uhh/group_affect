"""
.. moduleauthor:: Navin Raj Prabhu
"""


# -*- coding: UTF-8 -*-
import multiprocessing
import tempfile

import matplotlib.pyplot as plt
import pickle
import sys
from ctypes import c_char_p
import os
import csv
import numpy as np
import pandas as pd
import json


def standardize(signal):
    """
    It standardizes a monovariate/multivariate signals (in pandas DataFrame format) so that it has mean equal to zero and unitary variance.
    In case of a multivariate signal, standardization is carried out on each column of the DataFrame.
    
    :param signal:
        input signal
    :type signal: pd.DataFrame
    
    :returns: pd.DataFrame
            -- standardized signal
    """
    
    ' Raise error if parameters are not in the correct type '
    try :
        if not(isinstance(signal, pd.DataFrame)) : raise TypeError("Requires signal to be a pd.DataFrame")
    except TypeError as err_msg:
        raise TypeError(err_msg)
        return

    mean=signal.mean(axis=0)
    std=signal.std(axis=0)
    signal_norm=(signal-mean)/std

    return signal_norm
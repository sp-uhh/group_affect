import numpy as np

def gradient(data, axis=0):
    return np.amax(data, axis=axis) - np.amin(data, axis=axis)
"""
EE 298-F Machine Problem 2
Miranda, Neil Jon Louie P.
2007-46489
"""

import os.path as path
import struct

import numpy as np
from scipy import io as scipy_io

import mp2_util

def load_bin_file(filename, task):
    if not path.exists(filename):
        print('File %s not found.' % filename)
        return None

    if not task in mp2_util.DET_TASKS:
        print('Task %s not found.' % task)
        return None

    data = []
    raw_data = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    num_rows = raw_data.shape[0]
    for i in range(num_rows):
        x = raw_data[i, 0]
        y = raw_data[i, 1]
        z = raw_data[i, 2]

        if x >= mp2_util.MIN_X[task] and x <= mp2_util.MAX_X[task] \
                and y >= mp2_util.MIN_Y[task] and y <= mp2_util.MAX_Y[task] \
                and z >= mp2_util.MIN_Z[task] and z <= mp2_util.MAX_Z[task]:
            data.append(raw_data[i, :])

    return np.asarray(data)

"""
EE 298-F Machine Problem 2
Miranda, Neil Jon Louie P.
2007-46489
"""

import copy
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
    voxel_point_count = {}      # Contains number of points per voxel
    voxel_centroids = {}        # Contains centroid of each voxel

    raw_data = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    # np.random.shuffle(raw_data)

    num_rows = raw_data.shape[0]
    for i in range(num_rows):
        x = raw_data[i, 0]
        y = raw_data[i, 1]
        z = raw_data[i, 2]

        if x >= mp2_util.MIN_X[task] and x <= mp2_util.MAX_X[task] \
                and y >= mp2_util.MIN_Y[task] and y <= mp2_util.MAX_Y[task] \
                and z >= mp2_util.MIN_Z[task] and z <= mp2_util.MAX_Z[task]:
            coords = mp2_util.get_voxel_coordinates(raw_data[i, :], task)
            index = mp2_util.to_voxel_index(coords, task)

            if index in voxel_point_count:
                if voxel_point_count[index] < mp2_util.T[task]:
                    voxel_point_count[index] += 1
                    voxel_centroids[index] += raw_data[i, 0:3]
                    data.append(raw_data[i, :])
            else:
                voxel_point_count[index] = 1
                voxel_centroids[index] = 0 + raw_data[i, 0:3]
                data.append(raw_data[i, :])

    # Ignore all voxels with less than forty points
    voxel_point_count = {
        key : value for key, value in voxel_point_count.items() \
        if voxel_point_count[key] >= 40
    }

    voxel_centroids = {
        key : (value / voxel_point_count[key]) \
        for key, value in voxel_centroids.items() \
        if key in voxel_point_count.keys()
    }

    num_voxels = mp2_util.DIM_D[task] * mp2_util.DIM_H[task] * \
                 mp2_util.DIM_W[task]
    voxel_buffer = np.zeros((num_voxels, mp2_util.T[task], 7))
    voxel_feature_mask = np.zeros((num_voxels, mp2_util.T[task], 1))

    # Counter for sorting points into the voxel buffer
    point_ctr = {
        key : 0 for key in voxel_point_count.keys()
    }

    num_rows = len(data)
    for i in range(num_rows):
        coords = mp2_util.get_voxel_coordinates(data[i], task)
        index = mp2_util.to_voxel_index(coords, task)

        if not index in voxel_point_count:
            continue

        voxel_buffer[index, point_ctr[index], 0:4] = data[i]
        voxel_buffer[index, point_ctr[index], 4:] = \
            data[i][0:3] - voxel_centroids[index]
        point_ctr[index] += 1

    # Create feature mask which is multiplied after every VFE layer to ensure
    # that empty points in the voxel buffer do not contribute to weights
    for key, value in voxel_point_count.items():
        voxel_feature_mask[key, 0:value, 0] = 1

    return [voxel_buffer, voxel_feature_mask]

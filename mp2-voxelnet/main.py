"""
EE 298-F Machine Problem 2
Miranda, Neil Jon Louie P.
2007-46489
"""

import mp2_dataset

[voxel_buffer, voxel_feature_mask] = mp2_dataset.load_bin_file('dataset/training/velodyne/007479.bin', 'Car')
print(voxel_buffer.shape)
print(voxel_feature_mask.shape)

# trial = mp2_dataset.load_bin_file('dataset/training/velodyne/007480.bin', 'Cyclist')
# print(trial)
# print(trial.shape)

from keras.layers import Input

import mp2_model
import mp2_util

task = 'Car'

# input = Input(shape=(128, 10, 400, 352))
# input = Input(shape=(128, 5, 200, 176))

voxel_buffer = Input(shape=voxel_buffer.shape)
voxel_feature_mask = Input(shape=voxel_feature_mask.shape)
input = [voxel_buffer, voxel_feature_mask]

# input = Input(shape=(4000, 35, 7))

model = mp2_model.create_model(input, task)
model.summary()
print(mp2_model.get_model_memory_usage(2, model))

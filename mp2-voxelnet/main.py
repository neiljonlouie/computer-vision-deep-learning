"""
EE 298-F Machine Problem 2
Miranda, Neil Jon Louie P.
2007-46489
"""

import mp2_dataset

trial = mp2_dataset.load_bin_file('dataset/training/velodyne/007480.bin', 'Car')
print(trial)
print(trial.shape)

trial = mp2_dataset.load_bin_file('dataset/training/velodyne/007480.bin', 'Cyclist')
print(trial)
print(trial.shape)

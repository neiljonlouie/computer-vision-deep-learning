# EE 298-F Final Project
# Miranda, Neil Jon Louie P.
# 2007-46489

import os

import cv2
import keras
import numpy as np

class DatasetGenerator(keras.utils.Sequence):
    def __init__(self, dataset_root_dir, path_to_list_filename, batch_size=16):
        self.batch_size = batch_size

        list_filename = os.path.join(dataset_root_dir, path_to_list_filename)
        full_list_filename = os.path.abspath(list_filename)
        
        self.image_filenames = []
        self.label_filenames = []

        if not os.path.exists(full_list_filename):
            print ('File %s not found.' % self.full_list_filename)
            return

        list_file = open(full_list_filename, 'r')
        data = list(list_file)
        for line in data:
            strings = line.split()
            self.image_filenames.append(os.path.join(dataset_root_dir, strings[0]))
            self.label_filenames.append(os.path.join(dataset_root_dir, strings[1]))
        
    def __len__(self):
        return np.ceil(len(self.label_filenames) / float(self.batch_size))
        
    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.label_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]

        return np.array([cv2.imread(image_path) for image_path in batch_x]), \
               np.array([cv2.imread(label_path) for label_path in batch_y])

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
            self.image_filenames.append(os.path.normpath(dataset_root_dir + strings[0]))
            self.label_filenames.append(os.path.normpath(dataset_root_dir + strings[1]))
        
    def __len__(self):
        return np.int(np.ceil(len(self.label_filenames) / float(self.batch_size)))
        
    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.label_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]

        x = []
        y = []

        for i in range(len(batch_x)):
            image = cv2.imread(batch_x[i])
            label = cv2.imread(batch_y[i], cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, dsize=(1632,608))
            label = cv2.resize(label, dsize=(1632,608))

            label_flat = np.array(label, dtype=int).flatten()
            label_onehot = np.zeros((len(label_flat), 5))
            label_onehot[np.arange(len(label_flat)), label_flat] = 1
            label_onehot = np.reshape(label_onehot, (label.shape[0], label.shape[1], 5))

            x.append(image)
            y.append(label_onehot)

        return np.array(x), np.array(y, dtype=int)

# EE 298-F Final Project
# Miranda, Neil Jon Louie P.
# 2007-46489

from dataset_generator import DatasetGenerator as Generator

import cv2

dataset_root_dir = '/home/neil/Documents/culanedataset'
trn_generator = Generator(dataset_root_dir, 'list/train_gt.txt')
val_generator = Generator(dataset_root_dir, 'list/val_gt.txt')


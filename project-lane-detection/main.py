# EE 298-F Final Project
# Miranda, Neil Jon Louie P.
# 2007-46489

import dataset

import cv2

generator = dataset.preprocess_dataset('dataset', 'labels.json')

cv2.namedWindow('image')
# cv2.namedWindow('label')
for i in range(5):
    (image, label) = next(generator)
    cv2.imshow('image', image)
    # cv2.imshow('label', label)
    cv2.waitKey(1000)

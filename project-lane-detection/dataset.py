# EE 298-F Final Project
# Miranda, Neil Jon Louie P.
# 2007-46489

import json
import os

import cv2
import numpy as np

def preprocess_dataset(dir, label_filename):
    full_label_filename = os.path.join(dir, label_filename)
    if not os.path.exists(full_label_filename):
        print('File %s not found.' % full_label_filename)
        return None

    label_file = open(full_label_filename, 'r')
    labels = list(label_file)
    np.random.shuffle(labels)

    for line in labels:
        contents = json.loads(line)
        heights = contents['h_samples']
        file_name = contents['raw_file']
        image = cv2.imread(os.path.join(dir, file_name))
        image_height = image.shape[0]
        image_width = image.shape[1]
        label = np.zeros((image_height, image_width))

        for lane in contents['lanes']:
            zipped = list(zip(heights, lane))
            points = np.array([[width, height] for (height, width) in zipped \
                      if width > 0], dtype=np.int32)
            print(points)
            points = points.reshape(-1, 1, 2)
            cv2.polylines(label, [points], False, 1)

        yield (image, label)

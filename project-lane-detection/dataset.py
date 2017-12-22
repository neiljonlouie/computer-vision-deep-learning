# EE 298-F Final Project
# Miranda, Neil Jon Louie P.
# 2007-46489

from future_builtins import zip

import json
import os

import cv2

def preprocess_dataset(dir, label_filename):
    full_label_filename = os.path.join(dir, label_filename)
    if not os.path.exists(full_label_filename):
        print('File %s not found.' % full_label_filename)
        return None

    label_file = open(full_label_filename, 'r')
    for line in label_file:
        contents = json.loads(line)
        heights = contents['h_samples']
        for lane in contents['lanes']:
            zipped = list(zip(heights, lane))
            points = [(height, width) for (height, width) in zipped \
                      if width > 0]
            print(points)

        print('')

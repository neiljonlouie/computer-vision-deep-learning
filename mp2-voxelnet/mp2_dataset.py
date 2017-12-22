"""
EE 298-F Machine Problem 2
Miranda, Neil Jon Louie P.
2007-46489
"""

import csv, os, struct
import numpy as np
from scipy import io as scipy_io

import mp2_util

def load_bin_file(filename, task):
    if not os.path.exists(filename):
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

    num_voxels = mp2_util.DIM_X[task] * mp2_util.DIM_Y[task] * \
                 mp2_util.DIM_Z[task]
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


# Loads the contents of the label file, returns only the bounding boxes labeled
# with the specified task.
def load_label_file(filename, task):
    if not os.path.exists(filename):
        print('File %s not found.' % filename)
        return None

    if not task in mp2_util.DET_TASKS:
        print('Task %s not found.' % task)
        return None

    file = open(filename, 'r')
    file_reader = csv.reader(file, delimiter=' ')
    truths = []
    for row in file_reader:
        if row[0] == task:
            box = [float(i) for i in row[8:]]
            truths.append(box)

    return np.asarray(truths)


def compute_labels(boxes, task):
    output_height = int(mp2_util.DIM_X[task] // 2)
    output_width = int(mp2_util.DIM_Z[task] // 2)

    y_reg = np.zeros((output_height, output_width, 14))
    y_cls = np.zeros((output_height, output_width, 2))

    num_boxes = len(boxes)
    num_anchors_for_box = np.zeros(num_boxes).astype(int)
    best_anchor_for_box = -1 * np.ones((num_boxes, 7)).astype(int)
    best_iou_for_box = np.zeros(num_boxes)

    anchor_sizes = mp2_util.ANCHOR_SIZES[task]
    anchor_rotations = mp2_util.ANCHOR_ROTATIONS[task]
    anchor_y = mp2_util.ANCHOR_Y[task]

    height_spacing = float(mp2_util.MAX_X[task] - mp2_util.MIN_X[task]) / \
                           output_height
    width_spacing = float(mp2_util.MAX_Z[task] - mp2_util.MIN_Z[task]) / \
                          output_width
    x_coords = np.linspace(mp2_util.MIN_X[task] + 0.5 * height_spacing, \
                           mp2_util.MAX_X[task] - 0.5 * height_spacing, \
                           num=output_height)
    z_coords = np.linspace(mp2_util.MIN_Z[task] + 0.5 * width_spacing, \
                           mp2_util.MAX_Z[task] - 0.5 * width_spacing, \
                           num=output_width)

    print(boxes)
    for anchor_size in anchor_sizes:
        for anchor_rot in anchor_rotations:
            for ix in x_coords:
                for jz in z_coords:
                    anchor = [anchor_size[0], anchor_size[1], anchor_size[2], \
                              ix, anchor_y, jz, anchor_rot]
                    anchor_type = 'neg'
                    best_iou_for_loc = 0.0

                    for box_id in range(num_boxes):
                        curr_iou = mp2_util.compute_iou(boxes[box_id], anchor)
                        if curr_iou > 0:
                            print('%.3f' % curr_iou)


    return ([1, 2], [3, 4])


# Loads the dataset contained in the specified directory.
def load_dataset(data_dir, label_dir, task, batch_size):
    # Make sure that data_dir and label_dir exist and are directories
    if not os.path.isdir(data_dir):
        print('Error: %s is not a directory' % data_dir)
        return

    if not os.path.isdir(label_dir):
        print('Error: %s is not a directory' % label_dir)
        return

    # Prepare the list of files
    data_files = os.listdir(data_dir)
    label_files = os.listdir(label_dir)

    data_files = [file for file in data_files if file.endswith('.bin')]
    label_files = [file for file in label_files if file.endswith('.txt')]

    if len(label_files) != len(data_files):
        print('Error: The number of data files ("*.bin") must match '
              'the number of label files ("*.txt").')
        return

    data_files = np.sort(data_files)
    label_files = np.sort(label_files)

    i = 0
    num_files = len(data_files)
    count = 0
    while True:
        buffers = []
        features = []
        cls_labels = []
        reg_labels = []
        count = 0
        while count < batch_size:
            [buffer, feature] = load_bin_file(
                    os.path.join(data_dir, data_files[i]), task)
            buffers.append(buffer)
            features.append(feature)

            label = load_label_file(
                    os.path.join(label_dir, label_files[i]), task)
            cls_label, reg_label = compute_labels(label, task)
            cls_labels.append(cls_label)
            reg_labels.append(reg_label)

            i = (i + 1) % num_files
            count += 1

        buffers_stacked = np.stack(buffers)
        features_stacked = np.stack(features)
        cls_labels_stacked = np.stack(cls_labels)
        reg_labels_stacked = np.stack(reg_labels)

        yield ([np.copy(buffers_stacked), np.copy(features_stacked)], \
               [np.copy(cls_labels_stacked), np.copy(reg_labels_stacked)])

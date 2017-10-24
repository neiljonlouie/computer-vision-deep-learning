"""
EE 298-F Machine Problem 1
Miranda, Neil Jon Louie P.
2007-46489
"""

import random
import os

import cv2
import numpy as np

# Creates a dataset of image pairs, using JPEG files found in src_dir.
# Image pairs and resulting homography is stored in dst_dir.
def create_dataset(src_dir, dst_dir, res_height, res_width, patch_size,
                   perturb_size, batch_size=1):
    """Creates a dataset of image pairs from JPEG files found in src_dir.
    All images are resized before patches are extracted from them. The image
    pairs and resulting homography H_4pt are stored by batch in dst_dir.

    Args:
        src_dir (str): Directory containing images
        dst_dir (str): Directory where dataset is to be stored
        res_height (int): Height of resized image
        res_width (int): Width of resized image
        patch_size (int): Height and width of patch to be extracted from images
        perturb_size (int): Size of region where original corners are to be
                            perturbed
        batch_size (int): Number of images to be clumped into a single file
                          at the output

    Returns:
        None.
    """

    # Make sure that src_dir exists and is a directory
    if not os.path.isdir(src_dir):
        print('Error: %s is not a directory' % src_dir)
        return

    # If dst_dir does not exist yet, create the directory
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    # If dst_dir exists, make sure that it is a directory
    if not os.path.isdir(dst_dir):
        print('Error: %s is not a directory' % dst_dir)
        return

    # Perform checks on other parameters
    assert res_height > 0, 'Resized image height should be positive'
    assert res_width > 0, 'Resized image width should be positive'
    assert patch_size + 2 * perturb_size <= res_height, \
        'Patch width and perturbations should be within resized image height'
    assert patch_size + 2 * perturb_size <= res_width, \
        'Patch width and perturbations should be within resized image width'

    # Prepare the list of source images
    files = os.listdir(src_dir)
    files = np.sort(files)
    image_files = []
    for file in files:
        if file.endswith('.jpg'):
            image_files.append(file)

    len_image_files = len(image_files)
    if len_image_files == 0:
        print('No JPEG files found in %s' % src_dir)
        return
    else:
        print('JPEG files found in %s: %d' % (src_dir, len_image_files))

    num_batches = len_image_files / batch_size
    if num_batches == 0:
        print('Too few images (%d) to form one batch (%d).' %
              (len_image_files, batch_size))
        return

    print('Number of batches (of size %d) to be created: %d' %
          (batch_size, num_batches))
    idx = np.arange(0, len_image_files, batch_size)

    count = 0
    start = 1848
    for i in idx:
        batch_files = image_files[i : i + batch_size]
        if len(batch_files) < batch_size:
            break

        batch_data = []
        batch_labels = []
        for file in batch_files:
            src_img = cv2.imread(os.path.join(src_dir, file), 0)
            src_img_flipped = cv2.flip(src_img, 1)
            src_img_resized = cv2.resize(src_img_flipped, (res_width, res_height))

            # Image coordinates of first patch (unperturbed)
            # To make sure that bordering artifacts are not introduced, patch is
            # selected from the center of the resized image
            x1a = res_width / 2 - patch_size / 2
            y1a = res_height / 2 - patch_size / 2

            x2a = x1a + patch_size
            y2a = y1a

            x3a = x1a
            y3a = y1a + patch_size

            x4a = x1a + patch_size
            y4a = y1a + patch_size

            # Image coordinates of second patch (perturbed first)
            x1b = x1a + random.randint(-perturb_size, perturb_size)
            y1b = y1a + random.randint(-perturb_size, perturb_size)

            x2b = x2a + random.randint(-perturb_size, perturb_size)
            y2b = y2a + random.randint(-perturb_size, perturb_size)

            x3b = x3a + random.randint(-perturb_size, perturb_size)
            y3b = y3a + random.randint(-perturb_size, perturb_size)

            x4b = x4a + random.randint(-perturb_size, perturb_size)
            y4b = y4a + random.randint(-perturb_size, perturb_size)

            # Apply projective transformation on the image given the image
            # coordinates
            pts_a = np.float32([[x1a, y1a], [x2a, y2a], [x3a, y3a], [x4a, y4a]])
            pts_b = np.float32([[x1b, y1b], [x2b, y2b], [x3b, y3b], [x4b, y4b]])
            H = cv2.getPerspectiveTransform(pts_a, pts_b)
            H_inv = np.linalg.pinv(H)
            dst_img = cv2.warpPerspective(src_img_resized, H_inv,
                                          (res_width, res_height))

            # Save the pair as an array of shape (patch_size, patch_size, 2)
            dst_stacked = np.stack([src_img_resized[y1a:y4a, x1a:x4a],
                                   dst_img[y1a:y4a, x1a:x4a]], -1)

            # Save the 4-point homography as a list
            H_4point = np.array([
                            [x1b - x1a, y1b - y1a],
                            [x2b - x2a, y2b - y2a],
                            [x3b - x3a, y3b - y3a],
                            [x4b - x4a, y4b - y4a]
                       ])
            H_4point = np.ndarray.flatten(H_4point)

            batch_data.append(dst_stacked)
            batch_labels.append(H_4point)

        data = np.stack(batch_data)
        labels = np.stack(batch_labels)

        count += 1
        np.save(os.path.join(dst_dir, 'data_%05d.npy' % (count + start)), data)
        np.save(os.path.join(dst_dir, 'label_%05d.npy' % (count + start)), labels)

        if count % 10 == 0:
            print('%d/%d batches processed.' % (count, num_batches))

    print('Dataset created.')


# Loads the dataset contained in the specified directory.
# Assumes that data is stored in .npy files and their corresponding labels are
# stored in .txt files
def load_dataset(dataset_dir, test=False):
    # Make sure that dataset_dir exists and is a directory
    if not os.path.isdir(dataset_dir):
        print('Error: %s is not a directory' % dataset_dir)
        return

    # Prepare the list of files
    files = os.listdir(dataset_dir)
    data_files = []
    label_files = []
    for file in files:
        if (file.startswith('data_')):
            data_files.append(file)
        if (file.startswith('label_')):
            label_files.append(file)

    data_files = np.sort(data_files)
    label_files = np.sort(label_files)

    if len(label_files) != len(data_files):
        print('Error: The number of data files ("data_*") must match '
              'the number of label files("label_*").')
        return

    # data_list = []
    # labels_list = []
    i = 0
    num_batches = len(data_files)
    while True:
        data = np.load(os.path.join(dataset_dir, data_files[i]))
        label = np.load(os.path.join(dataset_dir, label_files[i]))
        i = (i + 1) % num_batches

        if test:
            label = label.astype('float32') / 2
            data = data[:, ::2, ::2, :]

        data = data.astype('float32') / 255
        yield (data, label)
    #     data_list.append(data)
    #     labels_list.append(label)
    #
    # data_all = np.stack(data_list)
    # labels_all = np.stack(labels_list)
    #
    # print(data_all.shape)
    # print(labels_all.shape)
    # return (data_all, labels_all)

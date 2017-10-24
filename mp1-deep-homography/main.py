"""
EE 298-F Machine Problem 1
Miranda, Neil Jon Louie P.
2007-46489
"""

from math import ceil
import os

from keras.callbacks import ModelCheckpoint, ProgbarLogger, TerminateOnNaN

import dataset
import model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

test_dir = 'test'
training_dir = 'train'

def generate_test_dataset():
    src_dir = 'test_raw'
    dst_dir = test_dir
    res_height = 480
    res_width = 640
    patch_width = 256
    perturb_size = 64
    batch_size = 64

    dataset.create_dataset(src_dir, dst_dir, res_height, res_width,
                           patch_width, perturb_size, batch_size)

def generate_training_dataset():
    src_dir = 'train_raw'
    dst_dir = training_dir
    res_width = 320
    res_height = 240
    patch_width = 128
    perturb_size = 32
    batch_size = 64

    dataset.create_dataset(src_dir, dst_dir, res_height, res_width,
                           patch_width, perturb_size, batch_size)

def train_regression_model():
    num_batches = 1848
    total_iterations = 90000
    batch_size = 64

    num_epochs = ceil(total_iterations / num_batches)
    initial_epoch = 0
    regression_model_file = 'model_%02d.hdf5' % initial_epoch

    regression_model = None
    if os.path.exists(regression_model_file):
        print('Loading from saved file.')
        regression_model = model.get_regression_model(regression_model_file)
    else:
        print('Start training from scratch.')
        regression_model = model.create_regression_model()
    regression_model.summary()

    progbar = ProgbarLogger('steps')
    checkpoint = ModelCheckpoint('model_{epoch:02d}.hdf5', verbose=1,
                                 monitor='loss')
    terminate = TerminateOnNaN()
    callbacks = [checkpoint, progbar, terminate]

    regression_model.fit_generator(generator=dataset.load_dataset(training_dir),
                                   steps_per_epoch=num_batches,
                                   epochs=num_epochs,
                                   callbacks=callbacks,
                                   initial_epoch=initial_epoch,
                                   verbose=1)

def test_regression_model():
    regression_model_file = 'model.hdf5'
    if not os.path.exists(regression_model_file):
        print('Model file not found.')
        return

    regression_model = model.get_regression_model(regression_model_file)
    num_steps = 5

    score = regression_model.evaluate_generator(
        generator=dataset.load_dataset(test_dir, test=True),
        steps=num_steps)
    # print('Test mean average corner error: %.02f' % score)
    print(score)
    print(regression_model.metrics_names)


# generate_test_dataset()
generate_training_dataset()
# train_regression_model()
# test_regression_model()

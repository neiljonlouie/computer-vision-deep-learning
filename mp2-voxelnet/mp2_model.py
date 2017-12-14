"""
EE 298-F Machine Problem 2
Miranda, Neil Jon Louie P.
2007-46489
"""

import numpy as np

from keras import backend as K
from keras.layers import Activation, BatchNormalization, Dense
from keras.layers import Concatenate, Lambda, Multiply, Permute, Reshape
from keras.layers import Conv2D, Conv2DTranspose, Conv3D
from keras.layers import ZeroPadding2D, ZeroPadding3D
from keras.models import Model

import mp2_util

def concatenate_aggregated_features(x, max_points):
    x_max = K.max(x, axis=2, keepdims=True)
    x_max_repeat = K.repeat_elements(x_max, rep=max_points, axis=2)
    y = K.concatenate([x, x_max_repeat], axis=-1)

    return y


def expand_feature_mask(input):
    voxel_buffer = input[0]
    voxel_feature_mask = input[1]
    shape = K.int_shape(voxel_buffer)
    voxel_feature_mask_repeated = K.repeat_elements(voxel_feature_mask, \
            rep=shape[3], axis=3)
    return voxel_feature_mask_repeated


def create_convolution_block(input, num_output, kernel_size, stride, num_layers):
    y = Conv2D(filters=num_output, kernel_size=kernel_size, strides=stride, \
               padding='same', data_format='channels_first')(input)
    y = BatchNormalization(axis=1)(y)
    y = Activation(activation='relu')(y)

    for i in range(num_layers - 1):
        y = Conv2D(filters=num_output, kernel_size=kernel_size, strides=stride,\
                   padding='same', data_format='channels_first')(y)
        y = BatchNormalization(axis=1)(y)
        y = Activation(activation='relu')(y)

    return y


def create_feature_learning_network(input, task):
    voxel_buffer = input[0]
    voxel_feature_mask = input[1]

    max_points = mp2_util.T[task]
    shape = K.int_shape(voxel_buffer)
    y = Reshape(target_shape=(-1, shape[-1]))(voxel_buffer)
    y = Dense(16)(y)
    y = BatchNormalization(axis=-1)(y)
    y = Activation(activation='relu')(y)

    shape = K.int_shape(y)
    y = Reshape(target_shape=(-1, max_points, shape[-1]))(y)
    y = Lambda(concatenate_aggregated_features, \
               arguments={'max_points': max_points})(y)
    z = Lambda(expand_feature_mask)([y, voxel_feature_mask])
    y = Multiply()([y, z])

    shape = K.int_shape(y)
    y = Reshape(target_shape=(-1, shape[-1]))(y)
    y = Dense(64)(y)
    y = BatchNormalization(axis=-1)(y)
    y = Activation(activation='relu')(y)

    shape = K.int_shape(y)
    y = Reshape(target_shape=(-1, max_points, shape[-1]))(y)
    y = Lambda(concatenate_aggregated_features, \
               arguments={'max_points': max_points})(y)
    z = Lambda(expand_feature_mask)([y, voxel_feature_mask])
    y = Multiply()([y, z])

    y = Dense(128)(y)
    y = BatchNormalization(axis=-1)(y)
    y = Activation(activation='relu')(y)

    y = Lambda(K.max, arguments={'axis': 2, 'keepdims': True})(y)

    shape = K.int_shape(y)
    y = Reshape(target_shape=(mp2_util.DIM_D[task], mp2_util.DIM_H[task], \
                              mp2_util.DIM_W[task], shape[-1]))(y)
    y = Permute((4, 1, 2, 3))(y)

    return y


def create_middle_network(input):
    y = ZeroPadding3D(padding=1, data_format='channels_first')(input)
    y = Conv3D(filters=64, kernel_size=3, strides=(1, 1, 1), \
               data_format='channels_first')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation(activation='relu')(y)

    y = ZeroPadding3D(padding=(0, 1, 1), data_format='channels_first')(y)
    y = Conv3D(filters=64, kernel_size=3, strides=1, \
               data_format='channels_first')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation(activation='relu')(y)

    y = ZeroPadding3D(padding=1, data_format='channels_first')(y)
    y = Conv3D(filters=64, kernel_size=3, strides=(2, 1, 1), \
               data_format='channels_first')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation(activation='relu')(y)

    return y


def create_region_proposal_network(input, task):
    shape = K.int_shape(input)
    y = Reshape(target_shape=((-1,) + shape[-2:]))(input)

    y = ZeroPadding2D(padding=1, data_format='channels_first')(y)
    if task == 'Car':
        y = Conv2D(filters=128, kernel_size=3, strides=2, \
                   data_format='channels_first')(y)
    else:
        y = Conv2D(filters=128, kernel_size=3, strides=1, \
                   data_format='channels_first')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation(activation='relu')(y)
    deconv1_input = create_convolution_block(y, 128, 3, 1, 3)

    y = ZeroPadding2D(padding=1, data_format='channels_first')(deconv1_input)
    y = Conv2D(filters=128, kernel_size=3, strides=2, \
               data_format='channels_first')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation(activation='relu')(y)
    deconv2_input = create_convolution_block(y, 128, 3, 1, 5)

    y = ZeroPadding2D(padding=1, data_format='channels_first')(deconv2_input)
    y = Conv2D(filters=256, kernel_size=3, strides=2, \
               data_format='channels_first')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation(activation='relu')(y)
    deconv3_input = create_convolution_block(y, 256, 3, 1, 5)

    deconv1_output = Conv2DTranspose(filters=256, kernel_size=3, strides=1, \
            padding='same', data_format='channels_first')(deconv1_input)
    deconv2_output = Conv2DTranspose(filters=256, kernel_size=2, strides=2, \
            data_format='channels_first')(deconv2_input)
    deconv3_output = Conv2DTranspose(filters=256, kernel_size=4, strides=4, \
            data_format='channels_first')(deconv3_input)

    print(K.int_shape(deconv1_input))
    print(K.int_shape(deconv2_input))
    print(K.int_shape(deconv3_input))

    y = Concatenate(axis=1)([deconv1_output, deconv2_output, deconv3_output])

    cls_output = Conv2D(filters=2, kernel_size=1, strides=1, padding='valid', \
                        data_format='channels_first')(y)
    reg_output = Conv2D(filters=14, kernel_size=1, strides=1, padding='valid',\
                        data_format='channels_first')(y)

    return [cls_output, reg_output]


def create_model(input, task):
    feature_learning_network = create_feature_learning_network(input, task)
    middle_network = create_middle_network(feature_learning_network)
    # middle_network = create_middle_network(input)
    rpn_output = create_region_proposal_network(middle_network, task)
    model = Model(input, rpn_output)
    # model = Model(input, feature_learning_network)

    return model


def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

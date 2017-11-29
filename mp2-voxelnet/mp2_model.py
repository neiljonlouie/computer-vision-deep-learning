"""
EE 298-F Machine Problem 2
Miranda, Neil Jon Louie P.
2007-46489
"""

from keras import backend as K
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Conv3D, Reshape, ZeroPadding2D, ZeroPadding3D
from keras.models import Model

def create_convolution_block(input, num_output, kernel_size, stride, num_layers):
    y = Conv2D(filters=num_output, kernel_size=kernel_size, strides=stride, padding='same', data_format='channels_first')(input)
    y = BatchNormalization(axis=1)(y)
    y = Activation(activation='relu')(y)

    for i in range(num_layers - 1):
        y = Conv2D(filters=num_output, kernel_size=kernel_size, strides=stride, padding='same', data_format='channels_first')(y)
        y = BatchNormalization(axis=1)(y)
        y = Activation(activation='relu')(y)

    return y


def create_middle_network(input):
    y = ZeroPadding3D(padding=1, data_format='channels_first')(input)
    y = Conv3D(filters=64, kernel_size=3, strides=(2, 1, 1), data_format='channels_first')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation(activation='relu')(y)

    y = ZeroPadding3D(padding=(0, 1, 1), data_format='channels_first')(y)
    y = Conv3D(filters=64, kernel_size=3, strides=1, data_format='channels_first')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation(activation='relu')(y)

    y = ZeroPadding3D(padding=1, data_format='channels_first')(y)
    y = Conv3D(filters=64, kernel_size=3, strides=(2, 1, 1), data_format='channels_first')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation(activation='relu')(y)

    return y


def create_region_proposal_network(input, task):
    shape = K.int_shape(input)
    y = Reshape(target_shape=((-1,) + shape[-2:]))(input)

    y = ZeroPadding2D(padding=1, data_format='channels_first')(y)
    if task == 'Car':
        y = Conv2D(filters=128, kernel_size=3, strides=2, data_format='channels_first')(y)
    else:
        y = Conv2D(filters=128, kernel_size=3, strides=1, data_format='channels_first')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation(activation='relu')(y)
    deconv1_input = create_convolution_block(y, 128, 3, 1, 3)

    y = ZeroPadding2D(padding=1, data_format='channels_first')(deconv1_input)
    y = Conv2D(filters=128, kernel_size=3, strides=2, data_format='channels_first')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation(activation='relu')(y)
    deconv2_input = create_convolution_block(y, 128, 3, 1, 5)

    y = ZeroPadding2D(padding=1, data_format='channels_first')(deconv2_input)
    y = Conv2D(filters=256, kernel_size=3, strides=2, data_format='channels_first')(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation(activation='relu')(y)
    deconv3_input = create_convolution_block(y, 256, 3, 1, 5)

    deconv1_output = Conv2DTranspose(filters=256, kernel_size=3, strides=1, padding='same', data_format='channels_first')(deconv1_input)
    deconv2_output = Conv2DTranspose(filters=256, kernel_size=2, strides=2, data_format='channels_first')(deconv2_input)
    deconv3_output = Conv2DTranspose(filters=256, kernel_size=4, strides=4, data_format='channels_first')(deconv3_input)

    y = Concatenate(axis=1)([deconv1_output, deconv2_output, deconv3_output])

    cls_output = Conv2D(filters=2, kernel_size=1, strides=1, padding='valid', data_format='channels_first')(y)
    reg_output = Conv2D(filters=14, kernel_size=1, strides=1, padding='valid', data_format='channels_first')(y)

    return [cls_output, reg_output]


def create_model(input, task):
    middle_network = create_middle_network(input)
    rpn_output = create_region_proposal_network(middle_network, task)
    model = Model(input, rpn_output)

    return model

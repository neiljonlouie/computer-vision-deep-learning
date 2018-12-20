# EE 298-F Final Project
# Miranda, Neil Jon Louie P.
# 2007-46489

from keras.models import Model
from keras.layers import Activation, Add, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

def build_enc_bn_relu_block(input):
    norm = BatchNormalization(axis=3)(input)
    return Activation('relu')(norm)


def build_enc_conv_bn_relu_block(filters, kernel_size, strides=(1,1), padding='same'):
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer='he_normal', kernel_regularizer=l2(1.e-4))(input)
        return build_enc_bn_relu_block(conv)
    
    return f


def build_enc_bn_relu_conv_block(filters, kernel_size, strides=(1,1), padding='same'):
    def f(input):
        bn_relu = build_enc_bn_relu_block(input)
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer='he_normal', kernel_regularizer=l2(1.e-4))(bn_relu)
        return conv
    
    return f


def enc_add_layers(input, res):
    input_shape = K.int_shape(input)
    res_shape = K.int_shape(res)
    stride_width = round(input_shape[2] / res_shape[2])
    stride_height = round(input_shape[1] / res_shape[1])

    if stride_width > 1 or stride_height > 1 or input_shape[3] != res_shape[3]:
        input = Conv2D(filters=res_shape[3], kernel_size=(1,1),
                       strides=(stride_width, stride_height), padding='valid',
                       kernel_initializer='he_normal', kernel_regularizer=l2(1.e-4))(input)
    
    return Add()([input, res])


def build_enc_residual_block(filters, is_first_layer=False, reduce_size=False):
    def f(input):
        if is_first_layer:
            conv_1 = Conv2D(filters=filters, kernel_size=(3,3),
                            strides=(1,1), padding='same',
                            kernel_initializer='he_normal', kernel_regularizer=l2(1.e-4))(input)
        else:
            if reduce_size:
                strides=(2,2)
            else:
                strides=(1,1)
            conv_1 = build_enc_bn_relu_conv_block(filters=filters, kernel_size=(3,3), strides=strides)(input)
        
        residual_1 = build_enc_bn_relu_conv_block(filters=filters, kernel_size=(3,3), strides=(1,1))(conv_1)
        out_1 = enc_add_layers(input, residual_1)

        conv_2 = build_enc_bn_relu_conv_block(filters=filters, kernel_size=(3,3), strides=(1,1))(out_1)
        residual_2 = build_enc_bn_relu_conv_block(filters=filters, kernel_size=(3,3), strides=(1,1))(conv_2)
        return enc_add_layers(out_1, residual_2)

    return f

def build_dec_residual_block(filters):
    def f(input):
        deconv = Conv2DTranspose(filters=filters, kernel_size=(3,3), strides=(2,2), padding='same')(input)
        input = UpSampling2D(size=(2,2))(input)
        input = Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1))(input)
        return Add()([input, deconv])

    return f

def build_model(input_shape):
    input = Input(shape=input_shape)
    conv_1 = build_enc_conv_bn_relu_block(filters=32, kernel_size=(7,7), strides=(2,2))(input)
    pool_1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv_1)

    enc_block_1 = build_enc_residual_block(filters=32, is_first_layer=True)(pool_1)
    enc_block_1 = build_enc_residual_block(filters=32)(enc_block_1)

    enc_block_2 = build_enc_residual_block(filters=64, reduce_size=True)(enc_block_1)
    enc_block_2 = build_enc_residual_block(filters=64)(enc_block_2)
    
    enc_block_3 = build_enc_residual_block(filters=128, reduce_size=True)(enc_block_2)
    enc_block_3 = build_enc_residual_block(filters=128)(enc_block_3)

    enc_block_4 = build_enc_residual_block(filters=256, reduce_size=True)(enc_block_3)
    enc_block_4 = build_enc_residual_block(filters=256)(enc_block_4)

    dec_block_1 = build_dec_residual_block(filters=128)(enc_block_4)
    dec_block_2 = build_dec_residual_block(filters=64)(dec_block_1)
    dec_block_3 = build_dec_residual_block(filters=32)(dec_block_2)
    dec_block_4 = build_dec_residual_block(filters=16)(dec_block_3)

    pre_output = Conv2DTranspose(filters=5, kernel_size=(3,3), strides=(2,2), padding='same')(dec_block_4)
    output = Activation(activation='softmax')(pre_output)

    model = Model(inputs=input, outputs=output)
    return model

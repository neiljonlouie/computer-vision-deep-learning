"""
EE 298-F Machine Problem 1
Miranda, Neil Jon Louie P.
2007-46489
"""

import mp1_metrics as metrics

from keras import optimizers
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten
from keras.layers import Input, MaxPooling2D
from keras.models import Model, load_model

def create_base_model(input):
    kernel_size = 3
    dropout_rate = 0.5

    y = Conv2D(filters=64, kernel_size=kernel_size, activation='relu')(input)
    y = BatchNormalization()(y)
    y = Conv2D(filters=64, kernel_size=kernel_size, activation='relu')(y)
    y = BatchNormalization()(y)

    y = MaxPooling2D(pool_size=(2, 2), strides=2)(y)

    y = Conv2D(filters=64, kernel_size=kernel_size, activation='relu')(y)
    y = BatchNormalization()(y)
    y = Conv2D(filters=64, kernel_size=kernel_size, activation='relu')(y)
    y = BatchNormalization()(y)

    y = MaxPooling2D(pool_size=(2, 2), strides=2)(y)

    y = Conv2D(filters=128, kernel_size=kernel_size, activation='relu')(y)
    y = BatchNormalization()(y)
    y = Conv2D(filters=128, kernel_size=kernel_size, activation='relu')(y)
    y = BatchNormalization()(y)

    y = MaxPooling2D(pool_size=(2, 2), strides=2)(y)

    y = Conv2D(filters=128, kernel_size=kernel_size, activation='relu')(y)
    y = BatchNormalization()(y)
    y = Conv2D(filters=128, kernel_size=kernel_size, activation='relu')(y)
    y = BatchNormalization()(y)

    y = Flatten()(y)
    y = Dropout(dropout_rate)(y)

    y = Dense(1024, activation='relu')(y)
    y = Dropout(dropout_rate)(y)

    return y


def create_classification_model():
    input = Input(shape=(128,128,2))
    y = create_base_model(input)
    # TODO: Implement classification model


def create_regression_model():
    input = Input(shape=(128,128,2))
    y = create_base_model(input)
    output = Dense(8)(y)

    sgd = optimizers.SGD(lr=0.005, momentum=0.9)
    adagrad = optimizers.Adagrad(lr=0.005)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=adagrad, loss=metrics.euclidean_loss,
                  metrics=[metrics.mean_average_corner_error])
    return model


def get_regression_model(model_file):
    model = load_model(model_file, custom_objects={
                           'euclidean_loss': metrics.euclidean_loss,
                           'mean_average_corner_error': metrics.mean_average_corner_error
                       })
    return model

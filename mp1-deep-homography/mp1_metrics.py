# EE 298-F Machine Problem 1
# Miranda, Neil Jon Louie P.
# 2007-46489

import keras.backend as K

def euclidean_loss(y_true, y_pred):
    y_diff = K.square(y_pred - y_true)
    return 0.5 * y_diff

def mean_average_corner_error(y_true, y_pred):
    y_diff = K.square(y_pred - y_true)
    y_diff = K.reshape(y_diff, (-1, 4, 2))
    y_dist = K.sum(y_diff, -1)
    return K.mean(K.sqrt(y_dist))

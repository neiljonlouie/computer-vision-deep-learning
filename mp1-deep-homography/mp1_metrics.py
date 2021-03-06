"""
EE 298-F Machine Problem 1
Miranda, Neil Jon Louie P.
2007-46489
"""

import keras.backend as K

# Loss function used in training the regression model.
def euclidean_loss(y_true, y_pred):
    y_diff = K.square(y_pred - y_true)
    return 0.5 * y_diff

# Metric used to evaluate the performance of the networks in the paper.
# Defined as the average of the L2 distances between the ground truth corner
# positions and the estimated corner positions.
def mean_average_corner_error(y_true, y_pred):
    y_diff = K.square(y_pred - y_true)
    y_diff = K.reshape(y_diff, (-1, 4, 2))
    y_dist = K.sum(y_diff, -1)
    return K.mean(K.sqrt(y_dist))

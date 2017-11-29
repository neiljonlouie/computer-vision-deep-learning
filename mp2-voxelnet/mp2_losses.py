"""
EE 298-F Machine Problem 2
Miranda, Neil Jon Louie P.
2007-46489
"""

from keras import backend as K

def smooth_l1_loss(x):
    flag = K.cast(K.less(x, 1.0), 'float32')
    return flag * (0.5 * x * x) + (1 - flag) * (K.abs(x) - 0.5)

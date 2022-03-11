
from tensorflow.keras import backend as K
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops


#################################################################
def masked_loss_function(y_true, y_pred):
    """ cumpute loss ignoring labels according to y_true[none, :, :, 0] where 7 means no data area
    # Arguments
    # Returns
    """
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))

    mask = K.cast(tf.not_equal(y_true, 7), K.floatx())

    y_true = K.one_hot(tf.compat.v1.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[:-1], axis=-1)

# # manual computation of crossentropy
    epsilon_ = ops.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)

    return -tf.reduce_sum(tf.divide(math_ops.reduce_sum(y_true * math_ops.log(y_pred), -1), tf.reduce_sum(mask)) )

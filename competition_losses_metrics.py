from tensorflow.python.keras.losses import binary_crossentropy
import tensorflow.python.keras.backend as K
import tensorflow as tf

from config import conf

# METRICS
def dice_coef(y_true, y_pred, smooth=1.):
    """
    Dice coefficient only on masks labels:
    :param y_true:
    :param y_pred:
    :param smooth:
    :return:
    """
    # # for metrics, it's good to round predictions:
    # y_pred = K.round(y_pred)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_class_0(y_true, y_pred):
    return dice_coef(y_true[:, :, :, 0], y_pred[:, :, :, 0])


def dice_class_1(y_true, y_pred):
    return dice_coef(y_true[:, :, :, 1], y_pred[:, :, :, 1])


def dice_class_2(y_true, y_pred):
    return dice_coef(y_true[:, :, :, 2], y_pred[:, :, :, 2])


def dice_class_3(y_true, y_pred):
    return dice_coef(y_true[:, :, :, 3], y_pred[:, :, :, 3])
#
# def dice_coef_multiclass(y_true, y_pred, smooth=1):
#     """
#     Dice coefficient on multiple class at once
#     :param y_true:
#     :param y_pred:
#     :param smooth:
#     :return:
#     """
#     # y_true_f = K.flatten(y_true)
#     # y_pred_f = K.flatten(y_pred)
#     y_true = tf.reshape(y_true, tf.shape(y_pred))
#     y_pred = K.round(y_pred)
#     intersection = K.sum(y_true * y_pred, axis=(0, 1, 2))
#     nominator = 2. * intersection + smooth
#     denominator = K.sum(y_true, axis=(0, 1, 2)) + K.sum(y_pred, axis=(0, 1, 2)) + smooth
#     return nominator / denominator


# LOSSES
def dice_loss(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    return 1. - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def generalised_dice_loss(y_true,
                          y_pred,
                          type_weight='Uniform'):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017
    :param y_pred: the logits
    :param y_true: the segmentation ground truth
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """
    # n_el = tf.cast(K.prod(tf.shape(y_pred)), tf.float32)
    # those ops need the y_true not to be 0! Although you find nan loss and accuracy
    if type_weight == 'Square':
        weights_op = lambda x: 1. / (tf.math.pow(K.sum(x, axis=(0, 1, 2)), y=3) + K.epsilon())
    elif type_weight == 'Simple':
        weights_op = lambda x: 1. / (tf.reduce_sum(x, axis=(0, 1, 2)) + K.epsilon())
    elif type_weight == 'Uniform':
        weights_op = lambda x: 1.
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    # treat each class separately
    w = weights_op(y_true)
    numerator = y_true * y_pred
    numerator = w * tf.reduce_sum(numerator, (0, 1, 2))
    numerator = tf.reduce_sum(numerator)

    denominator = tf.reduce_sum(y_true, (0, 1, 2)) + tf.reduce_sum(y_pred, (0, 1, 2))
    denominator = w * denominator
    denominator = tf.reduce_sum(denominator)

    gen_dice_coef = numerator / denominator
    # generalised_dice_score = 2. * num / denom
    return 1. - 2. * gen_dice_coef


# Focal Tversky loss, brought to you by:  https://github.com/nabsabraham/focal-tversky-unet
def tversky(y_true, y_pred, smooth=K.epsilon()):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)


def focal_tversky_loss(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)


# def focal_loss(alpha=0.25, gamma=2):
#     def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
#         weight_a = alpha * (1 - y_pred) ** gamma * targets
#         weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
#
#         return (tf.math.log1p(K.exp(-K.abs(logits))) + K.relu(-logits)) * (weight_a + weight_b) + logits * weight_b
#
#     def loss(y_true, y_pred):
#         y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
#         logits = tf.log(y_pred / (1 - y_pred))
#
#         loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
#
#         # or reduce_sum and/or axis=-1
#         return tf.reduce_mean(loss)
#
#     return loss




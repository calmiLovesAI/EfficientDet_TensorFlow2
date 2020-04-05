import tensorflow as tf


def item_assignment(input_tensor, boolean_mask, value, axes):
    """
    Support item assignment in TensorFlow
    :param input_tensor: A Tensor
    :param boolean_mask: A Tensor, dtype: tf.bool
    :param value: scalar
    :param axes : A list of scalar, the axes that are not used for masking
    :return: A Tensor with the same dtype as input_tensor
    """
    mask = tf.dtypes.cast(x=boolean_mask, dtype=input_tensor.dtype)
    for axis in axes:
        mask = tf.expand_dims(input=mask, axis=axis)
    masked_tensor = input_tensor * mask
    masked_value = value * (1 - mask)
    return  masked_tensor + masked_value
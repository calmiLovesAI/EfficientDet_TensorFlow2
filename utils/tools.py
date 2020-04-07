import tensorflow as tf


def item_assignment(input_tensor, boolean_mask, value, axes):
    """
    Support item assignment for tf.Tensor
    :param input_tensor: A Tensor
    :param boolean_mask: A Tensor, dtype: tf.bool
    :param value: A scalar
    :param axes : A list of scalar or None, the axes that are not used for masking
    :return: A Tensor with the same dtype as input_tensor
    """
    mask = tf.dtypes.cast(x=boolean_mask, dtype=input_tensor.dtype)
    if axes:
        for axis in axes:
            mask = tf.expand_dims(input=mask, axis=axis)
    masked_tensor = input_tensor * (1 - mask)
    masked_value = value * mask
    assigned_tensor = masked_tensor + masked_value
    return assigned_tensor


def advanced_item_assignmnet(input_tensor, boolean_mask, value, target_elements, elements_axis):
    """
    Supports assignment of specific elements for tf.Tensor
    :param input_tensor: A Tensor
    :param boolean_mask: A Tensor, dtype: tf.bool
    :param value: A scalar
    :param target_elements: A Tensor, shape: (N,), which specifies the index of the element to be assigned.
    :param elements_axis: A scalar, the axis of specific elements
    :return:
    """
    target_elements = item_assignment(target_elements, ~boolean_mask, -1, None)
    mask = tf.one_hot(indices=tf.cast(target_elements, dtype=tf.int32),
                      depth=input_tensor.shape[elements_axis],
                      axis=-1,
                      dtype=tf.float32)
    assigned_tensor = input_tensor * (1 - mask) + value * mask
    return assigned_tensor
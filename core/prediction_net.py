import tensorflow as tf
from core.bifpn import ConvNormAct


class BoxClassPredict(tf.keras.layers.Layer):
    def __init__(self, filters, depth, num_classes, num_anchors):
        super(BoxClassPredict, self).__init__()
        self.num_classes = num_classes
        self.box_convs = []
        self.class_convs = []
        for i in range(depth):
            self.box_convs.append(ConvNormAct(filters=filters,
                                              kernel_size=(3, 3),
                                              strides=1,
                                              padding="same"))
            self.class_convs.append(ConvNormAct(filters=filters,
                                                kernel_size=(3, 3),
                                                strides=1,
                                                padding="same"))
        self.box_head = tf.keras.layers.Conv2D(filters=num_anchors * 4,
                                               kernel_size=(3, 3),
                                               strides=1,
                                               padding="same")
        self.class_head = tf.keras.layers.Conv2D(filters=num_classes * num_anchors,
                                                 kernel_size=(3, 3),
                                                 strides=1,
                                                 padding="same")

    def call_single_level(self, inputs):
        box_feature = inputs
        class_feature = inputs
        for box_conv in self.box_convs:
            box_feature = box_conv(box_feature)
        for class_conv in self.class_convs:
            class_feature = class_conv(class_feature)
        box_pred = self.box_head(box_feature)
        box_pred = tf.reshape(tensor=box_pred, shape=(box_pred.shape[0], -1, 4))
        class_pred = self.class_head(class_feature)
        class_pred = tf.nn.sigmoid(class_pred)
        class_pred = tf.reshape(tensor=class_pred, shape=(class_pred.shape[0], -1, self.num_classes))
        return box_pred, class_pred

    def call(self, inputs, **kwargs):
        box_pred_levels = []
        class_pred_levels = []
        for x in inputs:
            box_pred, class_pred = self.call_single_level(x)
            box_pred_levels.append(box_pred)
            class_pred_levels.append(class_pred)
        box_preds = tf.concat(values=box_pred_levels, axis=1)
        class_preds = tf.concat(values=class_pred_levels, axis=1)
        return tuple([box_preds, class_preds])


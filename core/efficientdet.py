import tensorflow as tf
import numpy as np
from configuration import Config
from core.anchor import Anchors
from core.efficientnet import get_efficient_net
from core.bifpn import BiFPN
from core.loss import FocalLoss
from core.prediction_net import BoxClassPredict


class EfficientDet(tf.keras.Model):
    def __init__(self):
        super(EfficientDet, self).__init__()
        self.backbone = get_efficient_net(width_coefficient=Config.get_width_coefficient(),
                                          depth_coefficient=Config.get_depth_coefficient(),
                                          dropout_rate=Config.get_dropout_rate())
        self.bifpn = BiFPN(output_channels=Config.get_w_bifpn(), layers=Config.get_d_bifpn())
        self.prediction_net = BoxClassPredict(filters=Config.get_w_bifpn(),
                                              depth=Config.get_d_class(),
                                              num_classes=Config.num_classes,
                                              num_anchors=Config.num_anchor_per_pixel)

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: 4-D tensor, shape: (N, H, W, C)
        :param training:
        :param mask:
        :return: x: tuple, (box_preds, class_preds)
        """
        x = self.backbone(inputs, training=training)
        x = self.bifpn(x, training=training)
        x = self.prediction_net(x, training=training)
        return x


class PostProcessing:
    def __init__(self):
        self.anchors = Anchors(scales=Config.scales, ratios=Config.ratios, image_size=Config.get_image_size())

    def training_procedure(self, efficientdet_ouputs, labels):
        loss = FocalLoss()
        anchors = self.anchors(efficientdet_ouputs)
        reg_results, cls_results = efficientdet_ouputs
        cls_loss_value, reg_loss_value = loss(cls_results, reg_results, anchors, labels)
        loss_value = tf.math.reduce_mean(cls_loss_value) + tf.reduce_mean(reg_loss_value)
        # loss_value = np.mean(cls_loss_value) + np.mean(reg_loss_value)
        # loss_value = tf.convert_to_tensor(value=loss_value, dtype=tf.dtypes.float32)
        return loss_value

    def testing_procedure(self):
        pass

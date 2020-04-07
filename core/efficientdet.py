import tensorflow as tf
import numpy as np

from configuration import Config
from core.anchor import Anchors
from core.efficientnet import get_efficient_net
from core.bifpn import BiFPN
from core.loss import FocalLoss
from core.prediction_net import BoxClassPredict
from utils.nms import NMS


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
        self.loss = FocalLoss()

    def training_procedure(self, efficientdet_ouputs, labels):
        anchors = self.anchors(efficientdet_ouputs)
        reg_results, cls_results = efficientdet_ouputs[..., :4], efficientdet_ouputs[..., 4:]
        cls_loss_value, reg_loss_value = self.loss(cls_results, reg_results, anchors, labels)
        loss_value = tf.math.reduce_mean(cls_loss_value) + tf.reduce_mean(reg_loss_value)
        return loss_value

    def testing_procedure(self, efficientdet_ouputs, input_image_size):
        box_transform = BoxTransform()
        clip_boxes = ClipBoxes()
        map_to_original = MapToInputImage(input_image_size)
        nms = NMS()

        anchors = self.anchors(efficientdet_ouputs)
        reg_results, cls_results = efficientdet_ouputs[..., :4], efficientdet_ouputs[..., 4:]

        transformed_anchors = box_transform(anchors, reg_results)
        transformed_anchors = clip_boxes(transformed_anchors)
        transformed_anchors = map_to_original(transformed_anchors)
        scores = tf.math.reduce_max(cls_results, axis=2).numpy()
        classes = tf.math.argmax(cls_results, axis=2).numpy()
        final_boxes, final_scores, final_classes = nms(boxes=transformed_anchors[0, :, :],
                                                       box_scores=np.squeeze(scores),
                                                       box_classes=np.squeeze(classes))
        return final_boxes.numpy(), final_scores.numpy(), final_classes.numpy()



class BoxTransform:

    def __call__(self, boxes, deltas, *args, **kwargs):
        deltas = deltas.numpy()
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        center_x = boxes[:, :, 0] + 0.5 * widths
        center_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * 0.1
        dy = deltas[:, :, 1] * 0.1
        dw = deltas[:, :, 2] * 0.2
        dh = deltas[:, :, 3] * 0.2

        pred_center_x = center_x + dx * widths
        pred_center_y = center_y + dy * heights
        pred_w = np.exp(dw) * widths
        pred_h = np.exp(dh) * heights

        pred_boxes_x1 = pred_center_x - 0.5 * pred_w
        pred_boxes_y1 = pred_center_y - 0.5 * pred_h
        pred_boxes_x2 = pred_center_x + 0.5 * pred_w
        pred_boxes_y2 = pred_center_y + 0.5 * pred_h

        pred_boxes = np.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], axis=2)

        return pred_boxes


class ClipBoxes:
    def __init__(self):
        self.height, self.width = Config.get_image_size()[0], Config.get_image_size()[1]

    def __call__(self, boxes, *args, **kwargs):
        boxes[:, :, 0] = np.clip(a=boxes[:, :, 0], a_min=0, a_max=None)
        boxes[:, :, 1] = np.clip(a=boxes[:, :, 1], a_min=0, a_max=None)
        boxes[:, :, 2] = np.clip(a=boxes[:, :, 2], a_min=self.width, a_max=None)
        boxes[:, :, 3] = np.clip(a=boxes[:, :, 3], a_min=self.height, a_max=None)
        return boxes


class MapToInputImage:
    def __init__(self, input_image_size):
        self.h, self.w = input_image_size
        self.x_ratio = self.w / Config.get_image_size()[1]
        self.y_ratio = self.h / Config.get_image_size()[0]

    def __call__(self, boxes, *args, **kwargs):
        boxes[:, :, 0] = boxes[:, :, 0] * self.x_ratio
        boxes[:, :, 1] = boxes[:, :, 1] * self.y_ratio
        boxes[:, :, 2] = boxes[:, :, 2] * self.x_ratio
        boxes[:, :, 3] = boxes[:, :, 3] * self.y_ratio
        return boxes

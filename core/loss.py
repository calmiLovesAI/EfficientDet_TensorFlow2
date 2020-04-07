import tensorflow as tf
import numpy as np

from utils.iou import IOU
from configuration import Config
from utils.tools import item_assignment


class FocalLoss:
    def __init__(self):
        self.alpha = Config.alpha
        self.gamma = Config.gamma

    def __call__(self, cls_results, reg_results, anchors, labels):
        assert cls_results.shape[0] == reg_results.shape[0]
        batch_size = cls_results.shape[0]
        cls_loss_list = []
        reg_loss_list = []

        anchor = anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_center_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_center_y = anchor[:, 1] + 0.5 * anchor_heights

        for n in range(batch_size):
            class_result = cls_results[n, :, :]
            reg_result = reg_results[n, :, :]

            box_annotation = labels[n, :, :]
            # Filter out the extra padding boxes.
            box_annotation = box_annotation[box_annotation[:, 4] != -1]

            if box_annotation.shape[0] == 0:
                cls_loss_list.append(tf.constant(0, dtype=tf.dtypes.float32))
                reg_loss_list.append(tf.constant(0, dtype=tf.dtypes.float32))
                continue

            class_result = tf.clip_by_value(t=class_result, clip_value_min=1e-4, clip_value_max=1.0 - 1e-4)

            iou_value = IOU(box_1=anchor, box_2=box_annotation[:, :4]).calculate_iou()
            iou_max = tf.math.reduce_max(iou_value, axis=1)
            iou_argmax = tf.math.argmax(iou_value, axis=1)

            targets = tf.ones_like(class_result) * -1
            # targets = item_assignment(input_tensor=targets,
            #                           boolean_mask=tf.math.less(iou_max, 0.4),
            #                           value=0,
            #                           axes=[1])
            targets_numpy = targets.numpy()
            targets_numpy[np.less(iou_max.numpy(), 0.4), :] = 0
            targets = tf.convert_to_tensor(targets_numpy, dtype=tf.float32)

            positive_indices = tf.math.greater(iou_max, 0.5)
            num_positive_anchors = tf.reduce_sum(tf.dtypes.cast(x=positive_indices, dtype=tf.int32))
            assigned_annotations = box_annotation[iou_argmax, :]

            # targets = item_assignment(input_tensor=targets,
            #                           boolean_mask=positive_indices,
            #                           value=0,
            #                           axes=[1])
            targets_numpy = targets.numpy()
            targets_numpy[np.greater(iou_max.numpy(), 0.5), :] = 0
            targets = tf.convert_to_tensor(targets_numpy, dtype=tf.float32)

            targets_numpy = targets.numpy()
            targets_numpy[positive_indices, assigned_annotations[positive_indices, 4].astype(np.int)] = 1
            targets = tf.convert_to_tensor(targets_numpy, dtype=tf.float32)

            alpha_factor = tf.ones_like(targets) * self.alpha
            alpha_factor = tf.where(tf.math.equal(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = tf.where(tf.math.equal(targets, 1.), 1. - class_result, class_result)
            focal_weight = alpha_factor * tf.math.pow(focal_weight, self.gamma)
            bce = -(targets * tf.math.log(class_result) + (1.0 - targets) * tf.math.log(1.0 - class_result))

            cls_loss = focal_weight * bce
            cls_loss = tf.where(tf.math.not_equal(targets, -1.0), cls_loss, tf.zeros_like(cls_loss))
            cls_loss_list.append(tf.math.reduce_sum(cls_loss) / tf.keras.backend.clip(x=tf.cast(num_positive_anchors, dtype=tf.float32), min_value=1.0, max_value=None))

            if num_positive_anchors > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_center_x_pi = anchor_center_x[positive_indices]
                anchor_center_y_pi = anchor_center_y[positive_indices]
                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_center_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_center_y = assigned_annotations[:, 1] + 0.5 * gt_heights
                gt_widths = tf.keras.backend.clip(x=gt_widths, min_value=1, max_value=None)
                gt_heights = tf.keras.backend.clip(x=gt_heights, min_value=1, max_value=None)

                targets_dx = (gt_center_x - anchor_center_x_pi) / anchor_widths_pi
                targets_dy = (gt_center_y - anchor_center_y_pi) / anchor_heights_pi
                targets_dw = tf.math.log(gt_widths / anchor_widths_pi)
                targets_dh = tf.math.log(gt_heights / anchor_heights_pi)
                targets = tf.stack([targets_dx, targets_dy, targets_dw, targets_dh])
                targets = tf.transpose(a=targets, perm=[1, 0])
                targets = targets / tf.constant([[0.1, 0.1, 0.2, 0.2]])

                reg_diff = tf.math.abs(targets - tf.boolean_mask(reg_result, positive_indices, axis=0))
                reg_loss = tf.where(tf.math.less_equal(reg_diff, 1.0 / 9.0), 0.5 * 9.0 * tf.math.pow(reg_diff, 2), reg_diff - 0.5 / 9.0)
                reg_loss_list.append(tf.reduce_mean(reg_loss))
            else:
                reg_loss_list.append(tf.constant(0, dtype=tf.float32))

        final_cls_loss = tf.math.reduce_mean(tf.stack(cls_loss_list, axis=0), axis=0, keepdims=True)
        final_reg_loss = tf.math.reduce_mean(tf.stack(reg_loss_list, axis=0), axis=0, keepdims=True)

        return final_cls_loss, final_reg_loss

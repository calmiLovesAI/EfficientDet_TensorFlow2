import numpy as np

from utils.iou import IOU


class FocalLoss:
    def __init__(self):
        self.alpha = 0.25
        self.gamma = 2.0

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
            class_result = cls_results[n, :, :].numpy()
            reg_result = reg_results[n, :, :].numpy()

            box_annotation = labels[n, :, :]
            # Filter out the extra padding boxes.
            box_annotation = box_annotation[box_annotation[:, 4] != -1]

            if box_annotation.shape[0] == 0:
                cls_loss_list.append(np.array(0, dtype=np.float32))
                reg_loss_list.append(np.array(0, dtype=np.float32))
                continue

            class_result = np.clip(a=class_result, a_min=1e-4, a_max=1.0 - 1e-4)

            iou_value = IOU(box_1=anchor, box_2=box_annotation[:, :4]).calculate_iou()
            iou_max = np.max(iou_value, axis=1)
            iou_argmax = np.argmax(iou_value, axis=1)

            targets = np.ones_like(class_result)
            targets[np.less(iou_max, 0.4), :] = 0

            positive_indices = np.greater(iou_max, 0.5)
            num_positive_anchors = np.sum(positive_indices)
            assigned_annotations = box_annotation[iou_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].astype(np.int)] = 1

            alpha_factor = np.ones_like(targets) * self.alpha
            alpha_factor = np.where(np.equal(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = np.where(np.equal(targets, 1.), 1. - class_result, class_result)
            focal_weight = alpha_factor * np.power(focal_weight, self.gamma)
            bce = -(targets * np.log(class_result) + (1.0 - targets) * np.log(1.0 - class_result))

            cls_loss = focal_weight * bce
            cls_loss = np.where(np.not_equal(targets, -1.0), cls_loss, np.zeros_like(cls_loss))
            cls_loss_list.append(np.sum(cls_loss) / np.clip(a=num_positive_anchors.astype(np.float32), a_min=1.0, a_max=None))

            if np.sum(positive_indices) > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_center_x_pi = anchor_center_x[positive_indices]
                anchor_center_y_pi = anchor_center_y[positive_indices]
                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_center_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_center_y = assigned_annotations[:, 1] + 0.5 * gt_heights
                gt_widths = np.clip(a=gt_widths, a_min=1, a_max=None)
                gt_heights = np.clip(a=gt_heights, a_min=1, a_max=None)

                targets_dx = (gt_center_x - anchor_center_x_pi) / anchor_widths_pi
                targets_dy = (gt_center_y - anchor_center_y_pi) / anchor_heights_pi
                targets_dw = np.log(gt_widths / anchor_widths_pi)
                targets_dh = np.log(gt_heights / anchor_heights_pi)
                targets = np.stack([targets_dx, targets_dy, targets_dw, targets_dh])
                targets = targets.transpose()
                targets = targets / np.array([[0.1, 0.1, 0.2, 0.2]])

                reg_diff = np.abs(targets - reg_result[positive_indices, :])
                reg_loss = np.where(np.less_equal(reg_diff, 1.0 / 9.0), 0.5 * 9.0 * np.power(reg_diff, 2), reg_diff - 0.5 / 9.0)
                reg_loss_list.append(np.mean(reg_loss))
            else:
                reg_loss_list.append(np.array(0).astype(np.float32))

        final_cls_loss = np.mean(np.stack(cls_loss_list, axis=0), axis=0, keepdims=True)
        final_reg_loss = np.mean(np.stack(reg_loss_list, axis=0), axis=0, keepdims=True)

        return final_cls_loss, final_reg_loss

import tensorflow as tf

from configuration import Config


class NMS:
    def __init__(self):
        self.max_box_num = Config.max_box_num
        self.score_threshold = Config.score_threshold
        self.iou_threshold = Config.iou_threshold

    def __call__(self, boxes, box_scores, box_classes, *args, **kwargs):
        """
        :param boxes: A 2-D float Tensor(or numpy.ndarray) of shape [num_boxes, 4]
        :param box_scores: A 1-D float Tensor(or numpy.ndarray) of shape [num_boxes]
        :param box_classes: A 1-D float Tensor(or numpy.ndarray) of shape [num_boxes]
        :param args:
        :param kwargs:
        :return:
        """
        selected_indices = tf.image.non_max_suppression(boxes=boxes,
                                                        scores=box_scores,
                                                        max_output_size=self.max_box_num,
                                                        iou_threshold=self.iou_threshold,
                                                        score_threshold=self.score_threshold)
        selected_boxes = tf.gather(boxes, selected_indices)
        selected_scores = tf.gather(box_scores, selected_indices)
        selected_classes = tf.gather(box_classes, selected_indices)

        return selected_boxes, selected_scores, selected_classes


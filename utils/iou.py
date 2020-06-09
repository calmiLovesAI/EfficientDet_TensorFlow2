import numpy as np


class IOU:
    def __init__(self, box_1, box_2):
        self.box_1 = np.expand_dims(box_1, axis=1)
        self.box_2 = np.expand_dims(box_2, axis=0)

    @staticmethod
    def __get_box_area(box):
        return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])

    def calculate_iou(self):
        box_1_area = self.__get_box_area(self.box_1)
        box_2_area = self.__get_box_area(self.box_2)
        intersect_min = np.maximum(self.box_1[..., 0:2], self.box_2[..., 0:2])
        intersect_max = np.minimum(self.box_1[..., 2:4], self.box_2[..., 2:4])
        intersect_wh = np.maximum(intersect_max - intersect_min, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = box_1_area + box_2_area - intersect_area
        iou = intersect_area / union_area
        return iou
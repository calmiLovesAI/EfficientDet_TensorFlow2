import tensorflow as tf


class FocalLoss:
    def __init__(self):
        self.alpha = 0.25
        self.gamma = 2.0

    def __call__(self, cls_results, reg_results, anchors, labels):
        assert cls_results.shape[0] == reg_results.shape[0]
        batch_size = cls_results.shape[0]
        pass
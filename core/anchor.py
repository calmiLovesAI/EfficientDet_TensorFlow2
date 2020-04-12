import numpy as np
from configuration import Config


class Anchors:
    def __init__(self, scales, ratios, levels=5):
        self.scales = np.array(scales)
        self.ratios = np.array(ratios)
        self.levels = levels
        self.num_anchors = Config.num_anchor_per_pixel
        self.sizes = Config.sizes
        self.strides = Config.downsampling_strides

    def __call__(self, image_size, *args, **kwargs):
        image_size = np.array(image_size)
        image_shapes = [(image_size + s - 1) // s for s in self.strides]

        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for i in range(self.levels):
            anchors = self.__generate_anchors(size=self.sizes[i])   # shape: (self.num_anchors, 4)
            shifted_anchors = self.__shift(shape=image_shapes[i], stride=self.strides[i], anchors=anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
        all_anchors = np.expand_dims(all_anchors, axis=0)   # shape : (1, N, 4)
        return all_anchors.astype(np.float32)

    def __generate_anchors(self, size):
        assert self.num_anchors == len(self.scales) * len(self.ratios)
        anchors = np.zeros((self.num_anchors, 4))
        anchors[:, 2:] = size * np.tile(self.scales, (2, len(self.ratios))).T

        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]

        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(self.ratios, len(self.scales)))   # w
        anchors[:, 3] = anchors[:, 2] * np.repeat(self.ratios, len(self.scales))    # h

        # (center_x, center_y, w, h) ---> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T    # x1 and x2
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T    # y1 and y2

        return anchors

    def __shift(self, shape, stride, anchors):
        shift_x = (np.arange(0, shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, shape[0]) + 0.5) * stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel(),
            shift_x.ravel(), shift_y.ravel()
        )).transpose()

        A = anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = (anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))

        return all_anchors

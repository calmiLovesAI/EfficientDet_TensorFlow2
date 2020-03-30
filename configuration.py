

class Config:
    epochs = 50
    batch_size = 8

    network_type = "D0"

    # image size: (height, width)
    image_size = {"D0": (512, 512)}
    image_channels = 3

    # bifpn channels
    w_bifpn = {"D0": 64}
    # bifpn layers
    d_bifpn = {"D0": 2}
    # box/class layers
    d_class = {"D0": 3}

    def __init__(self):
        pass

    @classmethod
    def get_image_size(cls):
        return cls.image_size[cls.network_type]

    @classmethod
    def get_w_bifpn(cls):
        return cls.w_bifpn[cls.network_type]

    @classmethod
    def get_d_bifpn(cls):
        return cls.d_bifpn[cls.network_type]

    @classmethod
    def get_d_class(cls):
        return cls.d_class[cls.network_type]



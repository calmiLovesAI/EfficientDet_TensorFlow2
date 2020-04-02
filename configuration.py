

class Config:
    epochs = 50
    batch_size = 8

    network_type = "D0"

    # image size: (height, width)
    image_size = {"D0": (512, 512)}
    image_channels = 3

    # efficientnet
    width_coefficient = {"D0": 1.0, "D1": 1.0, "D2": 1.1, "D3": 1.2, "D4": 1.4, "D5": 1.6, "D6": 1.8, "D7": 1.8}
    depth_coefficient = {"D0": 1.0, "D1": 1.1, "D2": 1.2, "D3": 1.4, "D4": 1.8, "D5": 2.2, "D6": 2.6, "D7": 2.6}
    dropout_rate = {"D0": 0.2, "D1": 0.2, "D2": 0.3, "D3": 0.3, "D4": 0.4, "D5": 0.4, "D6": 0.5, "D7": 0.5}

    # bifpn channels
    w_bifpn = {"D0": 64}
    # bifpn layers
    d_bifpn = {"D0": 2}
    # box/class layers
    d_class = {"D0": 3}


    @classmethod
    def get_image_size(cls):
        return cls.image_size[cls.network_type]

    @classmethod
    def get_width_coefficient(cls):
        return cls.width_coefficient[cls.network_type]

    @classmethod
    def get_depth_coefficient(cls):
        return cls.depth_coefficient[cls.network_type]

    @classmethod
    def get_dropout_rate(cls):
        return cls.dropout_rate[cls.network_type]

    @classmethod
    def get_w_bifpn(cls):
        return cls.w_bifpn[cls.network_type]

    @classmethod
    def get_d_bifpn(cls):
        return cls.d_bifpn[cls.network_type]

    @classmethod
    def get_d_class(cls):
        return cls.d_class[cls.network_type]





class Config:
    epochs = 50
    batch_size = 8
    learning_rate_decay_epochs = 20

    # save model
    save_model_dir = "saved_model/"
    load_weights_before_training = False
    load_weights_from_epoch = 0
    save_frequency = 5

    network_type = "D0"

    # image size: (height, width)
    image_size = {"D0": (512, 512), "D1": (640, 640), "D2": (768, 768), "D3": (896, 896), "D4": (1024, 1024),
                  "D5": (1280, 1280), "D6": (1408, 1408), "D7": (1536, 1536)}
    image_channels = 3

    # efficientnet
    width_coefficient = {"D0": 1.0, "D1": 1.0, "D2": 1.1, "D3": 1.2, "D4": 1.4, "D5": 1.6, "D6": 1.8, "D7": 1.8}
    depth_coefficient = {"D0": 1.0, "D1": 1.1, "D2": 1.2, "D3": 1.4, "D4": 1.8, "D5": 2.2, "D6": 2.6, "D7": 2.6}
    dropout_rate = {"D0": 0.2, "D1": 0.2, "D2": 0.3, "D3": 0.3, "D4": 0.4, "D5": 0.4, "D6": 0.5, "D7": 0.5}

    # bifpn channels
    w_bifpn = {"D0": 64, "D1": 88, "D2": 112, "D3": 160, "D4": 224, "D5": 288, "D6": 384, "D7": 384}
    # bifpn layers
    d_bifpn = {"D0": 2, "D1": 3, "D2": 4, "D3": 5, "D4": 6, "D5": 7, "D6": 8, "D7": 8}
    # box/class layers
    d_class = {"D0": 3, "D1": 3, "D2": 3, "D3": 4, "D4": 4, "D5": 4, "D6": 5, "D7": 5}

    # nms
    score_threshold = 0.01
    iou_threshold = 0.5
    max_box_num = 100

    # dataset
    num_classes = 20
    pascal_voc_root = "./data/datasets/VOCdevkit/VOC2012/"
    pascal_voc_classes = {"person": 0, "bird": 1, "cat": 2, "cow": 3, "dog": 4,
                          "horse": 5, "sheep": 6, "aeroplane": 7, "bicycle": 8,
                          "boat": 9, "bus": 10, "car": 11, "motorbike": 12,
                          "train": 13, "bottle": 14, "chair": 15, "diningtable": 16,
                          "pottedplant": 17, "sofa": 18, "tvmonitor": 19}
    max_boxes_per_image = 20
    resize_mode = "RESIZE"

    # test image
    test_image_dir = "test_pictures/2007_000032.jpg"

    # txt file
    txt_file_dir = "data.txt"

    # anchors
    num_anchor_per_pixel = 9
    ratios = [0.5, 1, 2]
    scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    downsampling_strides = [8, 16, 16, 32, 32]

    # focal loss
    alpha = 0.25
    gamma = 2.0


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



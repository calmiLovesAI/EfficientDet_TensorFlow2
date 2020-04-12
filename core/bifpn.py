import tensorflow as tf


class BiFPN(tf.keras.layers.Layer):
    def __init__(self, output_channels, layers):
        super(BiFPN, self).__init__()
        self.levels = 5
        self.output_channels = output_channels
        self.layers = layers
        self.transform_convs = []
        self.bifpn_modules = []
        for _ in range(self.levels):
            self.transform_convs.append(ConvNormAct(filters=output_channels,
                                                    kernel_size=(1, 1),
                                                    strides=1,
                                                    padding="same"))
        for _ in range(self.layers):
            self.bifpn_modules.append(BiFPNModule(self.output_channels))

    def call(self, inputs, training=None, **kwargs):
        """
        :param inputs: list of features
        :param training:
        :param kwargs:
        :return: list of features
        """
        assert len(inputs) == self.levels
        x = []
        for i in range(len(inputs)):
            x.append(self.transform_convs[i](inputs[i], training=training))
        for j in range(self.layers):
            x = self.bifpn_modules[j](x, training=training)
        return x


class BiFPNModule(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(BiFPNModule, self).__init__()
        self.w_fusion_list = []
        self.conv_list = []
        for i in range(8):
            self.w_fusion_list.append(WeightedFeatureFusion(out_channels))
        self.upsampling_1 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.upsampling_2 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.upsampling_3 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.upsampling_4 = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.maxpool_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.maxpool_4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

    def call(self, inputs, training=None, **kwargs):
        """
        :param inputs: list of features
        :param training:
        :param kwargs:
        :return:
        """
        assert len(inputs) == 5
        f3, f4, f5, f6, f7 = inputs
        f6_d = self.w_fusion_list[0]([f6, self.upsampling_1(f7)], training=training)
        f5_d = self.w_fusion_list[1]([f5, self.upsampling_2(f6_d)], training=training)
        f4_d = self.w_fusion_list[2]([f4, self.upsampling_3(f5_d)], training=training)

        f3_u = self.w_fusion_list[3]([f3, self.upsampling_4(f4_d)], training=training)
        f4_u = self.w_fusion_list[4]([f4, f4_d, self.maxpool_1(f3_u)], training=training)
        f5_u = self.w_fusion_list[5]([f5, f5_d, self.maxpool_2(f4_u)], training=training)
        f6_u = self.w_fusion_list[6]([f6, f6_d, self.maxpool_3(f5_u)], training=training)
        f7_u = self.w_fusion_list[7]([f7, self.maxpool_4(f6_u)], training=training)

        return [f3_u, f4_u, f5_u, f6_u, f7_u]


class SeparableConvNormAct(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 padding):
        super(SeparableConvNormAct, self).__init__()
        self.conv = tf.keras.layers.SeparableConv2D(filters=filters,
                                                    kernel_size=kernel_size,
                                                    strides=strides,
                                                    padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.swish(x)
        return x


class ConvNormAct(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 padding):
        super(ConvNormAct, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.swish(x)
        return x


class WeightedFeatureFusion(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(WeightedFeatureFusion, self).__init__()
        self.epsilon = 1e-4
        self.conv = SeparableConvNormAct(filters=out_channels, kernel_size=(3, 3), strides=1, padding="same")

    def build(self, input_shape):
        self.num_features = len(input_shape)
        assert self.num_features >= 2
        self.fusion_weights = self.add_weight(name="fusion_w",
                                              shape=(self.num_features, ),
                                              dtype=tf.dtypes.float32,
                                              initializer=tf.constant_initializer(value=1.0 / self.num_features),
                                              trainable=True)

    def call(self, inputs, training=None, **kwargs):
        """
        :param inputs: list of features
        :param kwargs:
        :return:
        """
        fusion_w = tf.nn.relu(self.fusion_weights)
        sum_features = []
        for i in range(self.num_features):
            sum_features.append(fusion_w[i] * inputs[i])
        output_feature = tf.reduce_sum(input_tensor=sum_features, axis=0) / (tf.reduce_sum(input_tensor=fusion_w) + self.epsilon)
        output_feature = self.conv(output_feature, training=training)
        return output_feature


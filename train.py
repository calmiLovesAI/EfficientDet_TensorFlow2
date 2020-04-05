import tensorflow as tf


from core.efficientdet import EfficientDet, PostProcessing
from data.dataloader import DetectionDataset, DataLoader
from configuration import Config


def print_model_summary(network):
    sample_inputs = tf.random.normal(shape=(Config.batch_size, Config.get_image_size()[0], Config.get_image_size()[1], Config.image_channels))
    sample_outputs = network(sample_inputs, training=True)
    network.summary()


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # dataset
    dataset = DetectionDataset()
    train_data, train_size = dataset.generate_datatset()
    data_loader = DataLoader()
    steps_per_epoch = tf.math.ceil(train_size / Config.batch_size)

    # model
    efficientdet = EfficientDet()
    print_model_summary(efficientdet)

    load_weights_from_epoch = Config.load_weights_from_epoch
    if Config.load_weights_before_training:
        efficientdet.load_weights(filepath=Config.save_model_dir+"epoch-{}".format(load_weights_from_epoch))
        print("Successfully load weights!")
    else:
        load_weights_from_epoch = -1

    post_process = PostProcessing()

    # optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,
                                                                 decay_steps=steps_per_epoch * Config.learning_rate_decay_epochs,
                                                                 decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


    # metrics
    loss_metric = tf.metrics.Mean()

    def train_step(batch_images, batch_labels):
        with tf.GradientTape() as tape:
            pred = efficientdet(batch_images, training=True)
            loss_value = post_process.training_procedure(pred, batch_labels)
        gradients = tape.gradient(target=loss_value, sources=efficientdet.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, efficientdet.trainable_variables))
        loss_metric.update_state(values=loss_value)

    for epoch in range(load_weights_from_epoch + 1, Config.epochs):
        for step, batch_data in enumerate(train_data):
            images, labels = data_loader.read_batch_data(batch_data)
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}".format(epoch,
                                                                   Config.epochs,
                                                                   step,
                                                                   steps_per_epoch,
                                                                   loss_metric.result()))
        loss_metric.reset_states()

        if epoch % Config.save_frequency == 0:
            efficientdet.save_weights(filepath=Config.save_model_dir+"epoch-{}".format(epoch), save_format="tf")

    efficientdet.save_weights(filepath=Config.save_model_dir + "saved_model", save_format="tf")

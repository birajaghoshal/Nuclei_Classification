import os
import time
import keras
import tensorflow as tf
from generator import ImageLoader
from sklearn.model_selection import train_test_split


class Autoencoder:
    def __init__(self, config):
        """ Initialiser for the Model class.
                :param config: A ArgumentParser object
                """

        # Resets the Tensorflow computational graph.
        tf.reset_default_graph()

        # Sets the class variables from the config file.
        self.input_shape = [config.input_height, config.input_width, config.input_channels]
        self.num_classes = config.num_classes
        self.verbose = config.verbose
        self.config = config

        # Creates the classification model
        self.model = self.create_model()
        self.log("Model has been created\n")
        self.log(self.model.summary())

    def log(self, message):
        """ Function to handle printing and logging of messages.
        :param message: String of message to be printed and logged.
        """

        if self.config.verbose:
            print(message)
        if self.config.log_file != '':
            print(message, file=open(self.config.log_file, 'a'))

    def create_model(self):
        inputs = keras.layers.Input(self.input_shape)
        model = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv1.1")(inputs)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv1.2")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.MaxPool2D(padding="Same")(model)

        model = keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2.1")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2.2")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.MaxPool2D(padding="same")(model)

        model = keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv3.1")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv3.2")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv3.3")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.MaxPool2D(padding="same")(model)

        model = keras.layers.Conv2D(258, (3, 3), padding="same", activation="relu")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.Conv2D(258, (3, 3), padding="same", activation="relu")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.Conv2D(258, (3, 3), padding="same", activation="relu")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.UpSampling2D(padding="same")(model)

        model = keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.UpSampling2D(padding="same")(model)

        model = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.UpSampling2D(padding="same")(model)
        outputs = keras.layers.Conv2D(3, (3, 3), padding="same", activation="sigmoid")(model)

        model = keras.Model(inputs, outputs)
        optimiser = keras.optimizers.Adadelta(lr=self.config.learning_rate,
                                              rho=self.config.rho,
                                              epsilon=self.config.epsilon,
                                              decay=self.config.decay)

        model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, data):
        """ The main training loop for the model.
        :param data: A dataset object.
        """

        def pre_process(image):
            return image.astype('float32')/255.

        class EarlyStop(keras.callbacks.Callback):
            def __init__(self, min_epochs=0, batch=5, target=1., save_path="model", log_fn=print):
                super().__init__()
                self.val_losses = []
                self.train_losses = []
                self.min_epochs = min_epochs
                self.batch = batch
                self.target = target
                self.start_time = time.clock()
                self.log_fn = log_fn
                self.save_path = save_path

            def on_epoch_end(self, epoch, logs=None):
                message = 'Epoch: ' + str(epoch + 1).zfill(4)
                message += ' Training Loss: {:.4f}'.format(logs.get('loss'))
                message += ' Validation Accuracy: {:.4f}'.format(logs.get('val_acc'))
                message += ' Validation Loss: {:.4f}'.format(logs.get('val_loss'))
                message += ' Time: {:.5f}s'.format(time.clock() - self.start_time)
                self.log_fn(message)
                self.val_losses.append(logs.get('val_loss'))
                self.train_losses.append(logs.get('loss'))
                if (epoch + 1) >= self.min_epochs:
                    g_loss = 100 * ((self.val_losses[-1] / min(self.val_losses[:-1])) - 1)
                    t_progress = 1000 * ((sum(self.train_losses[-(self.batch + 1):-1]) /
                                          (self.batch * min(self.train_losses[-(self.batch + 1):-1]))) - 1)
                    self.log_fn('Training Progress: {:.4}'.format(g_loss / t_progress))
                    if g_loss / t_progress > self.target:
                        self.log_fn('Stopped at epoch ' + str(epoch + 1))
                        self.model.stop_training = True
                    else:
                        if self.save_path != '':
                            if os.path.exists(self.save_path):
                                os.remove(self.save_path)
                            self.model.save_weights(self.save_path, overwrite=True)
                else:
                    if self.save_path != '':
                        if os.path.exists(self.save_path):
                            os.remove(self.save_path)
                        self.model.save_weights(self.save_path, overwrite=True)

        gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=pre_process)

        data_x, val_x, _, _ = train_test_split(data.data_x, data.data_y, test_size=self.config.val_per)

        train_gen = ImageLoader(data_x, data_x, self.config.data_dir, gen,
                                target_size=(27, 27), batch_size=self.config.batch_size)
        val_gen = ImageLoader(val_x, val_x, self.config.data_dir, gen,
                              target_size=(27, 27), shuffle=False)

        history = self.model.fit_generator(train_gen, verbose=0,
                                           epochs=self.config.auto_max_epochs,
                                           validation_data=val_gen,
                                           callbacks=[EarlyStop(self.config.min_epochs,
                                                                self.config.batch_epochs,
                                                                self.config.auto_threshold,
                                                                self.config.model_path,
                                                                self.log)],
                                           use_multiprocessing=True)

        self.log("Finished Training Autoencoder!")

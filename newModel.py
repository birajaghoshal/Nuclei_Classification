import os
import keras
import pickle
import numpy as np
import tensorflow as tf


class Model:
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
        self.log("Classification Model has been created\n")
        self.log(self.model.summary())

    def __copy__(self):
        """ Resets the model and returns a copy of the model.
        :return: A reset copy of the reset Model
        """

        tf.reset_default_graph()
        return Model(self.config)

    def log(self, message):
        """ Function to handle printing and logging of messages.
        :param message: String of message to be printed and logged.
        """

        if self.config.verbose:
            print(message)
        if self.config.log_file != '':
            print(message, file=open(self.config.log_file, 'a'))

    def create_model(self):
        """ Creates a CNN model for Classification.
        :return: A computational graph representing a CNN model for Classification.
        """

        # Alteration to Keras implementation of Dropout to be applied during prediction.
        class AlwaysDropout(keras.layers.Dropout):
            def call(self, inputs, training=None):
                if 0. < self.rate < 1.:
                    noise_shape = self._get_noise_shape(inputs)
                    return keras.backend.dropout(inputs, self.rate, noise_shape, seed=self.seed)
                return inputs

        # Block 1
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(padding='same'))
        if self.config.bayesian:
            model.add(AlwaysDropout(0.25))

        # Block 2
        model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(padding='same'))
        if self.config.bayesian:
            model.add(AlwaysDropout(0.25))

        # Block 3
        model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D(padding='same'))
        if self.config.bayesian:
            model.add(AlwaysDropout(0.25))

        # Block 4
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        if self.config.bayesian:
            model.add(AlwaysDropout(0.25))
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(self.num_classes, activation='softmax'))

        # Creates an optimiser object.
        optimiser = keras.optimizers.Adadelta(lr=self.config.learning_rate,
                                              rho=self.config.rho,
                                              epsilon=self.config.epsilon,
                                              decay=self.config.decay)

        # Creates the model with the optimiser and loss function.
        return model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, data, test=True, experiment="0"):
        """ The main training loop for the model.
        :param data: A dataset object.
        :param test: Boolean if the model should be tested.
        :return: Training metrics, accuracy, mean class accuray, recall, precision, f1-score and loss.
        """

        class EarlyStopping(keras.callbacks.Callback):
            def __init__(self, min_epochs=0, batch=5, target=1.):
                super().__init__()
                self.val_losses = []
                self.train_losses = []
                self.min_epochs = min_epochs
                self.batch = batch
                self.target = target

            def on_epoch_end(self, epoch, logs=None):
                self.val_losses.append(logs.get('val_loss'))
                self.train_losses.append(logs.get('loss'))
                if (epoch + 1) % self.batch == 0 and (epoch + 1) >= self.min_epochs:
                    g_loss = 100 * ((self.val_losses[-1] / min(self.val_losses[:-1])) - 1)
                    t_progress = 1000 * ((sum(self.train_losses[-(self.batch+1):-1]) /
                                          (2 * min(self.train_losses[--(self.batch+1):-1]))) - 1)
                    print('Training Progress: {:.4}'.format(g_loss / t_progress))
                    if g_loss / t_progress > self.target:
                        print('Stopped at epoch ' + str(epoch + 1))
                        self.model.stop_training = True


        def data_gen(dataset):
            iterator = dataset.make_one_shot_iterator()
            next_batch = iterator.get_next()
            while True:
                yield keras.backend.get_session().run(next_batch)

        # Gets the training, testing and validation dataset objects.
        train_data, test_data, val_data = data.get_datasets(self.config.batch_size, 1000)

        # Loads the existing weights to the model.
        if self.config.model_tuning and self.config.mode != 'supervised' and os.path.isdir(self.config.model_path):
            self.model.load(self.config.model_path + '/weights')
            self.log('Model Restored')

        history = self.model.fit_generator(data_gen(train_data), epochs=self.config.max_epochs,
                                 validation_data=data_gen(val_data),
                                 verbose=2 if self.config.verbose else 0,
                                 callbacks=[EarlyStopping(self.config.min_epochs,
                                                          self.config.batch_epochs,
                                                          self.config.training_threshold)])

        with open('/History/' + experiment, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        if test:
            predictions = []
            for i in range(self.config.bayesian_iterations):
                iterator = test_data.make_one_shot_iterator()
                next_batch = iterator.get_next()
                temp_predictions, labels = [], []
                for _ in len(int(np.ceil(len(self.test_y) / 1000))):
                    image_batch, label_batch = keras.backend.get_session().run(next_batch)
                    temp_predictions += self.model.predict_on_batch(image_batch).tolist()
                    labels += label_batch.tolist()
                predictions.append(temp_predictions)
            predictions = np.average(predictions, axis=0)


import os
import time
import keras
import numpy as np
import tensorflow as tf
from collections import Counter
from generator import ImageLoader
import sklearn.metrics as metrics


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
        model = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv1.1")(inputs)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv1.2")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.MaxPool2D(padding="Same")(model)
        if self.config.bayesian:
            model = keras.layers.Dropout(0.5)(model, training=True)

        model = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name="conv2.1")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name="conv2.2")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.MaxPool2D(padding='same')(model)
        if self.config.bayesian:
            model = keras.layers.Dropout(0.5)(model, training=True)

        model = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name="conv3.1")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name="conv3.2")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', name="conv3.3")(model)
        model = keras.layers.BatchNormalization()(model)
        model = keras.layers.MaxPool2D(padding='same')(model)
        if self.config.bayesian:
            model = keras.layers.Dropout(0.5)(model, training=True)

        model = keras.layers.Flatten()(model)
        model = keras.layers.Dense(1024, activation='relu', name="dense1")(model)
        model = keras.layers.BatchNormalization()(model)
        if self.config.bayesian:
            model = keras.layers.Dropout(0.5)(model, training=True)
        model = keras.layers.Dense(1024, activation='relu', name="dense2")(model)
        model = keras.layers.BatchNormalization()(model)
        outputs = keras.layers.Dense(self.num_classes, activation='softmax', name="output")(model)

        model = keras.Model(inputs, outputs)
        optimiser = keras.optimizers.Adadelta(lr=self.config.learning_rate,
                                              rho=self.config.rho,
                                              epsilon=self.config.epsilon,
                                              decay=self.config.decay)

        # Creates the model with the optimiser and loss function.
        model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, data, test=True, method=np.average):
        """ The main training loop for the model.
        :param data: A dataset object.
        :param test: Boolean if the model should be tested.
        :return: Training metrics, accuracy, mean class accuray, recall, precision, f1-score and loss.
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
                    t_progress = 1000 * ((sum(self.train_losses[-(self.batch+1):-1]) /
                                          (self.batch * min(self.train_losses[-(self.batch+1):-1]))) - 1)
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

        # Loads the existing Weights to the model.
        if self.config.model_tuning and self.config.mode != 'supervised' and os.path.exists(self.config.model_path):
            self.model.load_weights(self.config.model_path)
            self.log('Model Restored')

        gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=pre_process)
        train_x, train_y = data.get_training_data()
        val_x, val_y = data.sample_data(data.val_x, data.val_y)
        if data.pseudo_indices != []:
            self.log("Sampled Training Patches with Pseudo Labels: " + str(len(train_x)))
        else:
            self.log("Sampled Training Patches: " + str(len(train_x)))
        self.log("Sampled Validation Patches: " + str(len(val_x)))
        train_gen = ImageLoader(train_x, train_y, self.config.data_dir, gen,
                                target_size=(27, 27), batch_size=self.config.batch_size)
        val_gen = ImageLoader(val_x, val_y, self.config.data_dir, gen,
                                target_size=(27, 27), shuffle=False)

        if self.config.weighted_loss:
            counter = Counter(train_gen.classes)
            max_val = float(max(counter.values()))
            class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}
        else:
            class_weights = None

        history = self.model.fit_generator(train_gen, verbose=0,
                                           epochs=self.config.max_epochs,
                                           validation_data=val_gen,
                                           class_weight=class_weights,
                                           callbacks=[EarlyStop(self.config.min_epochs,
                                                      self.config.batch_epochs,
                                                      self.config.training_threshold if test else  self.config.bootstrap_threshold,
                                                      self.config.model_path if test else '',
                                                      self.log)],
                                           use_multiprocessing=True)

        if test:
            self.model.load_weights(self.config.model_path)
            test_x, test_y = data.sample_data(data.test_x, data.test_y)
            test_gen = ImageLoader(test_x, test_y, self.config.data_dir, gen,
                                   target_size=(27, 27), shuffle=False)
            predictions = []
            iterations = self.config.bayesian_iterations if self.config.bayesian else 1

            for i in range(iterations):
                predictions.append(self.model.predict_generator(test_gen, use_multiprocessing=True))
                self.log('Bayesian Iteration: ' + str(i+1))
            predictions = np.average(predictions, axis=0) if iterations > 1 else predictions[0]

            predicted_averages, predicted_labels, labels = [], [], []
            for i in range(0, len(predictions), self.config.sample_size):
                averages = method(predictions[i:(i + self.config.sample_size)], axis=0)
                predicted_averages.append(averages)
                predicted_labels.append(np.argmax(averages))
                labels.append(test_y[i])

            loss = metrics.log_loss(labels, predicted_averages, labels=[0, 1, 2, 3])
            recall = metrics.recall_score(labels, predicted_labels, average='micro')
            precision = metrics.precision_score(labels, predicted_labels, average='micro')
            f1_score = metrics.f1_score(labels, predicted_labels, average='micro')
            cmat = metrics.confusion_matrix(labels, predicted_labels)
            accuracy = np.mean(cmat.diagonal() / cmat.sum(axis=1))
            accuracy_score = metrics.accuracy_score(labels, predicted_labels)

            # Prints the calculated testing metrics.
            message = '\nModel trained with an Accuracy: {:.4f}'.format(accuracy_score)
            message += ' Mean Class Accuracy: {:.4f}'.format(accuracy)
            message += ' Recall: {:.4f}'.format(recall)
            message += ' Precision: {:.4f}'.format(precision)
            message += ' F1-Score: {:.4f}'.format(f1_score)
            message += ' Loss: {:.4f}'.format(loss)
            self.log(message)

            return accuracy_score, accuracy, recall, precision, f1_score, loss
        else:
            gen = keras.preprocessing.image.ImageDataGenerator()
            data_x, data_y = data.sample_data(data.data_x, data.data_y)
            data_gen = ImageLoader(data_x, data_y, self.config.data_dir, gen, target_size=(27, 27), shuffle=False)

            predictions = []
            iterations = self.config.bayesian_iterations if self.config.bayesian else 1

            for i in range(iterations):
                predictions.append(self.model.predict_generator(data_gen, use_multiprocessing=True))
                self.log('Bayesian Iteration: ' + str(i + 1))
            predictions = np.average(predictions, axis=0) if iterations > 1 else predictions[0]

            predicted_averages, labels = [], []
            for i in range(0, len(predictions), self.config.sample_size):
                predicted_averages.append(method(predictions[i:(i + self.config.sample_size)], axis=0))
                labels.append(data_y[i])

            return predicted_averages, labels

    def predict(self, data, method=np.average):
        """ Make cell predictions from the unlabelled dataset.
        :param data: A dataset object.
        :param method: A method for how to combine the predictions of each cell.
        :return: A list of predictions for each cell.
        """

        self.model.load_weights(self.config.model_path)
        gen = keras.preprocessing.image.ImageDataGenerator()
        data_x, data_y = data.sample_data(data.data_x, data.data_y)
        data_gen = ImageLoader(data_x, data_y, self.config.data_dir, gen, target_size=(27, 27), shuffle=False)

        predictions = []
        iterations = self.config.bayesian_iterations if self.config.bayesian else 1

        for i in range(iterations):
            predictions.append(self.model.predict_generator(data_gen, use_multiprocessing=True))
            self.log('Bayesian Iteration: ' + str(i + 1))
        predictions = np.average(predictions, axis=0) if iterations > 1 else predictions[0]

        predicted_averages, labels = [], []
        for i in range(0, len(predictions), self.config.sample_size):
            predicted_averages.append(method(predictions[i:(i + self.config.sample_size)], axis=0))
            labels.append(data_y[i])

        return predicted_averages, labels

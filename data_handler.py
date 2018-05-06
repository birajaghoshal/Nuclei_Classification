import random
import numpy as np
import tensorflow as tf
from collections import Counter


class DataHandler:
    def __init__(self, config):
        """ The initialiser for the DataHandler class.
        :param config: A ArgumentParser object.
        """

        # Creates the lists to store data.
        self.train_x, self.train_y = np.array([]), np.array([])
        self.test_x, self.test_y = np.array([]), np.array([])
        self.val_x, self.val_y = np.array([]), np.array([])
        self.data_x, self.data_y = np.array([]), np.array([])

        # Sets the class members.
        self.val_per = config.val_per
        self.verbose = config.verbose
        self.config = config

        # Loads the training data into the unannotated data stores.
        self.load_training_data(config.data_dir)
        self.load_testing_data(config.data_dir)

    def log(self, message):
        """ Method to handle printing and logging of messages.
        :param message: String of message to be printed and logged.
        """

        if self.config.verbose:
            print(message)
        if self.config.log_file != '':
            print(message, file=open(self.config.log_file, 'a'))

    def load_training_data(self, data_dir):
        """ Loads the training data to the unannotated lists.
        :param data_dir: The data directory.
        """

        values = np.load(data_dir + "Training/values.npy")
        self.data_x = np.array(values[0])
        self.data_x = [data_dir + "Training/" + i for i in self.data_x]
        self.data_y = values[1].astype(int)

    def load_testing_data(self, data_dir):
        """ Loads the testing data to the testing data lists.
        :param data_dir: The data directory.
        """

        values = np.load(data_dir + "Testing/values.npy")
        self.test_x = np.array(values[0])
        self.test_x = [data_dir + "Testing/" + i for i in self.test_x]
        self.test_y = values[1].astype(int)

    def balance(self, x_list, y_list):
        """ A method to balance a set of data.
        :param x_list: A list of data.
        :param y_list: A list of labels.
        :return: balanced x and y lists.
        """

        # TODO - make this work with cell patches
        balance = Counter(y_list)
        min_values = min(list(balance.values()))
        indices = []
        for c in range(self.config.num_classes):
            class_values = balance[c]
            indices.append(np.random.permutation([j for j, i in enumerate(y_list) if i == c])
                           [:class_values - min_values])
        x_list = np.array([i for j, i in enumerate(x_list) if j not in indices])
        y_list = np.array([i for j, i in enumerate(y_list) if j not in indices])
        return x_list, y_list

    def set_validation_set(self):
        """ Sets the validation set from the training data.
        """

        num_val = int((len(self.train_y) / self.config.cell_patches) * self.val_per)
        indices = []
        for _ in range(num_val):
            random_index = random.randint(0, int(len(self.train_y) / num_val)) * self.config.cell_patches
            indices += list(range(random_index, random_index + self.config.cell_patches))

        val_x = np.take(self.train_x, indices)
        val_y = np.take(self.train_y, indices)
        self.train_x = np.delete(self.train_x, indices)
        self.train_y = np.delete(self.train_y, indices)

        # val_x = np.array([i for j, i in enumerate(self.train_x) if j in indices])
        # val_y = np.array([i for j, i in enumerate(self.train_y) if j in indices])
        # self.data_x = np.delete(self.train_x, indices)
        # self.data_y = np.delete(self.train_y, indices)
        if self.config.combine.lower() == "add":
            self.val_x = np.append(self.val_x, val_x)
            self.val_y = np.append(self.val_y, val_y)
        elif self.config.combine.lower() == "replace":
            self.val_x = val_x
            self.val_y = val_y

    def all_data(self):
        """ Sets all data from the unlabelled data to the training set.
        """

        # Sets the unlabelled data to the training set.
        self.train_x = self.data_x
        self.train_y = self.data_y
        self.data_x = np.array([])
        self.data_y = np.array([])

        # Balances the training data.
        if self.config.balance:
            self.train_x, self.train_y = self.balance(self.train_x, self.train_y)

        # Sets the validation set.
        self.set_validation_set()

        # Logs the number of patches.
        self.class_weights = list(Counter(self.train_y).values())
        self.log("Training Patches: " + str(len(self.train_y)))
        self.log("Validation Patches: " + str(len(self.val_y)))

    def set_training_data(self, indices):
        """ Sets data from the unlabelled data to the training set.
        :param indices: A list of indices to be moved from unlabelled to training.
        """

        # Sets the full list of indices
        full_indices = []
        for index in indices:
            full_indices.append(list(range(index, index + self.config.cell_patches)))

        # Sets temparary lists to the data to be added.
        temp_x = np.take(self.data_x, full_indices)
        temp_y = np.take(self.data_y, full_indices)

        # Removes the data from the unannotated list.
        self.data_x = np.delete(self.data_x, full_indices)
        self.data_y = np.delete(self.data_y, full_indices)

        # Balances the data.
        if self.config.balance:
            temp_x, temp_y = self.balance(temp_x, temp_y)

        # Adds the data depending on specified method.
        if self.config.combine.lower() == "add":
            self.train_x = np.append(self.train_x, temp_x)
            self.train_y = np.append(self.train_y, temp_y)
        elif self.config.combine.lower() == "replace":
            self.train_x = temp_x
            self.train_y = temp_y

        # Sets the validation data.
        self.set_validation_set()

        # Logs the number of patches.
        self.class_weights = list(Counter(self.train_y).values())
        self.log("Training Patches: " + str(len(self.train_y)))
        self.log("Validation Patches: " + str(len(self.val_y)))

    def get_num_batches(self, train_batch_size, test_batch_size):
        """ The the number of the batches for each set of data.
        :param train_batch_size: The size of the training batches.
        :param test_batch_size: The size of the testing and validation batches.
        :return: The number of batches for the training, testing and validation sets.
        """

        num_train_batches = int(np.ceil(len(self.train_y) / train_batch_size))
        num_test_batches = int(np.ceil(len(self.test_y) / test_batch_size))
        num_val_batches = int(np.ceil(len(self.val_y) / test_batch_size))
        return num_train_batches, num_test_batches, num_val_batches

    def input_parser(self, image_path, label):
        """ Operation for the parser to dynamically load images in the dataset.
        :param image_path: THe file path to the image.
        :param label: The label for the image to be loaded.
        :return: An operation to load the image and create a one hot label.
        """

        # Encodes the label as one hot.
        one_hot_label = tf.one_hot(tf.to_int32(label), self.config.num_classes)

        # Reads and decodes the image file.
        image_file = tf.read_file(image_path)
        image = tf.image.decode_bmp(image_file)

        # Returens the image and label.
        return tf.cast(image, "float"), one_hot_label

    def get_datasets(self, train_batch_size, test_batch_size):
        """ Produces the dataset objects to be used by the model.
        :param train_batch_size: The size of the batches for training.
        :param test_batch_size: The size of the batches for testing and validation.
        :return: The datasets objects to dynamically load the data.
        """

        # Training set - Loads the data into tensors
        #                Sets the input_parser as the operation for loading the data.
        #                Splits and shuffles the data into batches.
        train_images = tf.constant(self.train_x)
        train_labels = tf.constant(self.train_y)
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        train_dataset = train_dataset.shuffle(1000)
        train_dataset = train_dataset.map(self.input_parser, num_parallel_calls=self.config.parallel_calls)
        train_dataset = train_dataset.prefetch(train_batch_size * self.config.prefetch)
        train_dataset = train_dataset.batch(train_batch_size).repeat(None)

        # Testing set - Loads the data into tensors
        #               Sets the input_parser as the operation for loading the data.
        #               Splits the data into batches.
        test_images = tf.constant(self.test_x)
        test_labels = tf.constant(self.test_y)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        test_dataset = test_dataset.map(self.input_parser, num_parallel_calls=self.config.parallel_calls)
        test_dataset = test_dataset.prefetch(test_batch_size * self.config.prefetch)
        test_dataset = test_dataset.batch(test_batch_size)

        # Validation set - Loads the data into tensors
        #                  Sets the input_parser as the operation for loading the data.
        #                  Splits and shuffles the data into batches.
        val_images = tf.constant(self.val_x)
        val_labels = tf.constant(self.val_y)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        val_dataset = val_dataset.map(self.input_parser, num_parallel_calls=self.config.parallel_calls)
        val_dataset = val_dataset.prefetch(test_batch_size * self.config.prefetch)
        val_dataset = val_dataset.batch(test_batch_size)

        # Returns the dataset objects.
        return train_dataset, test_dataset, val_dataset

    def get_num_prediction_batches(self, batch_size):
        """ The the number of the batches for the unlabelled set of data.
        :param batch_size: The size of the prediction batches.
        :return: The number of batches for the predictions sets.
        """

        return int(np.ceil(len(self.data_y) / batch_size))

    def predict_input_parser(self, image_path):
        """ Operation for the parser to dynamically load images in the dataset for prediction.
        :param image_path: THe file path to the image.
        :return: An operation to load the image and create a one hot label.
        """

        image_file = tf.read_file(image_path)
        image = tf.image.decode_image(image_file, channels=self.config.input_channels)

        return tf.cast(image, "float")

    def get_prediction_datasets(self, batch_size):
        """ Produces the dataset objects to be used by the model for predictions.
        :param batch_size: The size of the batches for predictions.
        :return: The datasets objects to dynamically load the data for prediction.
        """

        images = tf.constant(np.asarray(self.data_x))
        predict_dataset = tf.data.Dataset.from_tensor_slices((images))
        predict_dataset = predict_dataset.map(self.predict_input_parser, num_parallel_calls=1000).prefetch(1000)
        return predict_dataset.batch(batch_size)

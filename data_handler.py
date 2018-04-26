import random
import numpy as np
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
        self.data_x = values[0]
        self.data_y = values[1]

    def load_testing_data(self, data_dir):
        """ Loads the testing data to the testing data lists.
        :param data_dir: The data directory.
        """

        values = np.load(data_dir + "Testing/values.npy")
        self.test_x = values[0]
        self.test_y = values[1]

    def balance(self, x_list, y_list):
        """ A method to balance a set of data.
        :param x_list: A list of data.
        :param y_list: A list of labels.
        :return: balanced x and y lists.
        """
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
        num_val = int(len(self.train_y) * self.val_per)
        self.val_x, self.val_y = zip(*random.sample(list(zip(self.train_x, self.train_y)), num_val))
        sort_idx = np.asarray(self.train_x).argsort()
        indices = sort_idx[np.searchsorted(np.asarray(self.train_x), np.asarray(self.val_x), sorter=sort_idx)].tolist()
        self.train_x = np.delete(self.train_x, indices)
        self.train_y = np.delete(self.train_y, indices)

        # Logs the number of patches.
        self.class_weights = list(Counter(self.train_y).values())
        self.log('Training Patches: ' + str(len(self.train_y)))
        self.log('Validation Patches: ' + str(len(self.val_y)))

    def set_training_data(self, indices):
        """ Sets data from the unlabelled data to the training set.
        :param indices: A list of indices to be moved from unlabelled to training.
        """

        # Sets temparary lists to the data to be added.
        temp_x = np.array([i for j, i in enumerate(self.data_x) if j in indices])
        temp_y = np.array([i for j, i in enumerate(self.data_y) if j in indices])

        # Removes the data from the unannotated list.
        self.data_x = np.array([i for j, i in enumerate(self.data_x) if j not in indices])
        self.data_y = np.array([i for j, i in enumerate(self.data_y) if j not in indices])

        # Balances the data.
        if self.config.balance:
            temp_x, temp_y = self.balance(temp_x, temp_y)

        # Sets the validation data
        num_val = int(len(temp_y) * self.val_per)
        val_x, val_y = zip(*random.sample(list(zip(temp_x, temp_y)), num_val))
        sort_idx = np.asarray(temp_x).argsort()
        indices = sort_idx[np.searchsorted(np.asarray(temp_x), np.asarray(val_x), sorter=sort_idx)].tolist()
        temp_x = np.delete(temp_x, indices)
        temp_y = np.delete(temp_y, indices)

        # Adds the data depending on specified method.
        if self.config.combine.lower() == 'add':
            self.val_x = np.append(self.val_x, val_x)
            self.val_y = np.append(self.val_y, val_y)
            self.train_x = np.append(self.train_x, temp_x)
            self.train_y = np.append(self.train_y, temp_y)
        elif self.config.combine.lower() == 'replace':
            self.val_x = val_x
            self.val_y = val_y
            self.train_x = temp_x
            self.train_y = temp_y

        # Logs the number of patches.
        self.class_weights = list(Counter(self.train_y).values())
        self.log('Training Patches: ' + str(len(self.train_y)))
        self.log('Validation Patches: ' + str(len(self.val_y)))



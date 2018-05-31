import numpy as np
from collections import Counter
import sklearn.metrics as metrics


class DataHandler:
    def __init__(self, config, load_data=True):
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
        self.pseudo_indices = []

        # Loads the training data into the unannotated data stores.
        if load_data:
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
        self.data_x = np.array(values[:, 0])
        self.data_x = np.array(["Training/" + i for i in self.data_x])
        self.data_y = values[:, 1].astype(int)
        self.log("Loaded " + str(int(len(self.data_y) / self.config.cell_patches)) + " Unannotated Cells")

    def load_testing_data(self, data_dir):
        """ Loads the testing data to the testing data lists.
        :param data_dir: The data directory.
        """

        values = np.load(data_dir + "Testing/values.npy")
        self.test_x = np.array(values[:, 0])
        self.test_x = np.array(["Testing/" + i for i in self.test_x])
        self.test_y = values[:,1].astype(int)
        self.log("Loaded " + str(int(len(self.test_y) / self.config.cell_patches)) + " Testing Cells")

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

    def set_validation_set(self, x, y):
        """ Sets the validation set from the training data.
        """

        num_val = int((len(y) / self.config.cell_patches) * self.val_per)
        indices = []
        cell_indices = np.random.choice(list(range(len(y) // self.config.cell_patches)), num_val, False)
        for i in cell_indices:
            index = i * self.config.cell_patches
            indices += list(range(index, index + self.config.cell_patches))

        val_x = np.take(x, indices)
        val_y = np.take(y, indices)
        x = np.delete(x, indices)
        y = np.delete(y, indices)

        # val_x = np.array([i for j, i in enumerate(self.train_x) if j in indices])
        # val_y = np.array([i for j, i in enumerate(self.train_y) if j in indices])
        # self.data_x = np.delete(self.train_x, indices)
        # self.data_y = np.delete(self.train_y, indices)
        if self.config.combine.lower() == "add":
            self.val_x = np.append(self.val_x, val_x)
            self.val_y = np.append(self.val_y, val_y)#, axis=0) if len(self.val_y) != 0 else val_y
        elif self.config.combine.lower() == "replace":
            self.val_x = val_x
            self.val_y = val_y

        return x, y

    def all_data(self):
        """ Sets all data from the unlabelled data to the training set.
        """

        # Sets the unlabelled data to the training set.
        self.train_x = self.data_x
        self.train_y = self.data_y
        self.data_x = np.array([])
        self.data_y = np.array([])

        # Sets the validation set.
        self.train_x, self.train_y = self.set_validation_set(self.train_x, self.train_y)

        # Balances the training data.
        if self.config.balance:
            self.train_x, self.train_y = self.balance(self.train_x, self.train_y)

        # Logs the number of patches.
        self.log("Training Patches: " + str(len(self.train_y)))
        self.log("Validation Patches: " + str(len(self.val_y)))

    def set_training_data(self, indices):
        """ Sets data from the unlabelled data to the training set.
        :param indices: A list of indices to be moved from unlabelled to training.
        """

        # Sets the full list of indices
        full_indices = []
        for index in indices:
            index *= self.config.cell_patches
            full_indices += list(range(index, index + self.config.cell_patches))

        # Sets temparary lists to the data to be added.
        temp_x = np.take(self.data_x, full_indices)
        temp_y = np.take(self.data_y, full_indices)#, axis=0)

        # Removes the data from the unannotated list.
        self.data_x = np.delete(self.data_x, full_indices)
        self.data_y = np.delete(self.data_y, full_indices)#, axis=0)

        # Sets the validation data.
        temp_x, temp_y = self.set_validation_set(temp_x, temp_y)

        # Balances the data.
        if self.config.balance:
            temp_x, temp_y = self.balance(temp_x, temp_y)

        # Adds the data depending on specified method.
        if self.config.combine.lower() == "add":
            self.train_x = np.append(self.train_x, temp_x)
            self.train_y = np.append(self.train_y, temp_y)#, axis=0) if len(self.train_y) != 0 else temp_y
        elif self.config.combine.lower() == "replace":
            self.train_x = temp_x
            self.train_y = temp_y

        # Logs the number of patches.
        self.log("Training Patches: " + str(len(self.train_y)))
        self.log("Validation Patches: " + str(len(self.val_y)))

    def sample_data(self, x, y):
        """ Method for randomly sampling each cell within the inputted data.
        :param x: The x data.
        :param y: The y data.
        :return: Sampled x and y data.
        """

        indices = []
        for i in range(0, len(x) - 1, self.config.cell_patches):
            cell_indices = list(range(i, i + self.config.cell_patches))
            indices += np.random.choice(cell_indices, self.config.sample_size, replace=False).tolist()
        return np.take(x, indices), np.take(y, indices)

    def add_pesudo_labels(self, predictions, labels):
        """ Adds unlabelled cells to be used in training data.
        :param cell_indices: The indices of the cells.to be added as pesudo labels
        """

        indices = [j for j, i in enumerate(predictions) if max(i) > self.config.pseudo_threshold]
        self.pseudo_indices = []
        for cell_index in indices:
            index = cell_index * self.config.cell_patches
            self.pseudo_indices += list(range(index, index + self.config.cell_patches))
        self.log("Pesudo Cells: " + str(len(indices)))
        self.log("Pesudo Patches: " + str(len(indices) * self.config.cell_patches))
        predicted_labels = np.argmax(np.array(predictions)[indices], axis=1)
        self.log("Pesudo Accuracy: " + str(float(metrics.accuracy_score(np.array(labels)[indices], predicted_labels))))

    def get_training_data(self):
        """ Method for getting the data for training including pesudo labels and sampling.
        :return: Two lists representing x and y data.
        """

        if self.config.mode != "bootstrap":
            train_x = np.append(self.train_x, self.data_x[self.pseudo_indices])
            train_y = np.append(self.train_y, self.data_y[self.pseudo_indices])
            return train_x, train_y
        else:
            return self.train_x, self.train_y

    def get_bootstraps(self, data_x, data_y, shortlist_indices):
        """ Method for extracting bootstraped data handelers.
        :param data_x: A list of data
        :param data_y: A list of labels
        :return: A list of data handelers
        """

        bootstraps = []
        for _ in range(self.config.bootstrap_number):
            indices = np.random.choice(range(0, len(data_y // self.config.cell_patches), self.config.cell_patches),
                                       self.config.bootstrap_size, replace=True)
            full_indices = []
            for index in indices:
                full_indices += list(range(index, index + self.config.cell_patches))

            bootstrap_x = data_x[full_indices]
            bootstrap_y = data_y[full_indices]
            
            data = DataHandler(self.config, False)
            bootstrap_x, bootstrap_y = data.set_validation_set(bootstrap_x, bootstrap_y)
            data.train_x = bootstrap_x
            data.train_y = bootstrap_y
            full_indices = []
            for i in shortlist_indices:
                full_indices += list(range(i, i + self.config.cell_patches))
            data.data_x = self.data_x[full_indices]
            data.data_y = self.data_y[full_indices]
            bootstraps.append(data)
        return bootstraps


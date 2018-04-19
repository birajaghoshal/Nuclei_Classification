import numpy as np


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
        self.test_per = config.test_per
        self.verbose = config.verbose
        self.config = config

        # Loads the training data into the unannotated data stores.
        self.load_training_data(config.data_dir)

    def log(self, message):
        """ Method to handle printing and logging of messages.
        :param message: String of message to be printed and logged.
        """

        if self.config.verbose:
            print(message)
        if self.config.log_file != '':
            print(message, file=open(self.config.log_file, 'a'))

    def load_training_data(self, data_dir):
        values = np.load(data_dir + "values.npy")
        self.data_x = values[0]
        self.data_y = values[1]

    def load_testing_data(self, data_dir):
        values = np.load(data_dir + "Testing/values.npy")
        self.test_x = values[0]
        self.test_y = values[1]

        elif self.config.combine.lower() == 'replace':
            self.val_x = val_x
            self.val_y = val_y

        # indices = [j for j, i in enumerate(temp_x) if i in val_x]
        sort_idx = np.asarray(temp_x).argsort()
        indices = sort_idx[np.searchsorted(np.asarray(temp_x), np.asarray(val_x), sorter=sort_idx)].tolist()

        temp_x = np.delete(temp_x, indices).tolist()
        temp_y = np.delete(temp_y, indices).tolist()

        if self.config.balance:
            balance = Counter(temp_y)
            if balance[0] > balance[1]:
                # Gets the number of patches to remove.
                number = balance[0] - balance[1]

                # Randomly selects patches to be removed.
                indices = np.random.permutation([i for i, j in enumerate(temp_y) if j == 0])[:number]

            elif balance[0] < balance[1]:
                # Gets the number of patches to remove.
                number = balance[1] - balance[0]

                # Randomly selects patches to be removed.
                indices = np.random.permutation([i for i, j in enumerate(temp_y) if j == 1])[:number]

            temp_x = [i for j, i in enumerate(temp_x) if j not in indices]
            temp_y = [i for j, i in enumerate(temp_y) if j not in indices]
            temp_y = [i for j, i in enumerate(temp_y) if j not in indices]

        if self.config.combine.lower() == 'add':
            self.train_x += temp_x
            self.train_y += temp_y
        elif self.config.combine.lower() == 'replace':
            self.train_y = temp_y

        counter = list(Counter(self.train_y).items())
        self.class_weights = [m[1] for m in counter if True]

        self.log('Training Patches: ' + str(len(self.train_y)))
        self.log('Validation Patches: ' + str(len(self.val_y)))

import copy


class ActiveLearner:
    def __init__(self, data, model, config):
        """ Initialiser for the ActiveLearner Class.
        :param data: A DataHandler object.
        :param model: A Model object.
        :param config: A Config object.
        """

        self.data = data
        self.model = model
        self.config = config

    def log(self, message):
        """ Method to handle printing and logging of messages.
        :param message: String of message to be printed and logged.
        """

        if self.config.verbose:
            print(message)
        if self.config.log_file != '':
            print(message, file=open(self.config.log_file, 'a'))

    def train(self):
        """ Makes a new model and trains it using the data.
        :return: The testing metrics from training the CNN: Mean Class Accuracy, Recall, Precision, F1-Score and Loss.
        """

        model = copy.copy(self.model)
        return model.train(self.data)

    def predict(self, method):
        """ Makes a new model and makes predictions on the unlabelled data.
        :return: The predictions for each cell.
        """

        model = copy.copy(self.model)
        return model.predict(self.data, method)

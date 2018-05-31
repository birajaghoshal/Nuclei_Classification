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

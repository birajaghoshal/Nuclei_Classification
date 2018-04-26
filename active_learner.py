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

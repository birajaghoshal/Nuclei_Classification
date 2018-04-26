import config
from model import Model
from data_handler import DataHandler
from random_learner import Random


def log(config, message):
    """ Function to handle printing and logging of messages.
    :param config: An ArgumentParser object.
    :param message: String of message to be printed and logged.
    """
    if config.verbose:
        print(message)
    if config.log_file != '':
        print(message, file=open(config.log_file, 'a'))


if __name__ == "__main__":
    config = config.load_configs()
    if config.mode.lower() == 'supervised':
        model = Model(config)
        data_handler = DataHandler(config)
        data_handler.all_data()
        model.train(data_handler)
    elif config.mode.lower() == 'random':
        model = Model(config)
        data_handler = DataHandler(config)
        random_learner = Random(data_handler, model, config)
        random_learner.run()

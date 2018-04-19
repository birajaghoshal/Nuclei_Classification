import config
from model import Model
from data_handler import DataHandler


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
    model = Model(config)
    data = DataHandler(config)
    data.set_training_data(list(range(len(data.data_y))))

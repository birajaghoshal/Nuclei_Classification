import config


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

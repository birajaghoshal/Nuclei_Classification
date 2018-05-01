import config
from model import Model
import matplotlib.pyplot as plt
from random_learner import Random
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


def plotting(title, values, increment):
    """ Produces and saves figures with the learning curve.
    :param title: The title of the figure.
    :param values: The list of training metrics from the cycles.
    :param increment: The increment of training regions.
    """

    plt.plot(list(range(increment, increment * len(values) + increment, increment)), values)
    plt.xlabel('Patches')
    plt.ylabel(title)
    plt.title(title)
    plt.savefig(title + '.png')
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == "__main__":
    config = config.load_configs()
    if config.mode.lower() == 'supervised':
        model = Model(config)
        data_handler = DataHandler(config)
        data_handler.all_data()
        model.train(data_handler)

    elif config.mode.lower() in ['random']:
        model = Model(config)
        data_handler = DataHandler(config)

        if config.mode.lower() == 'random':
            learner = Random(data_handler, model, config)

        accuracies, mean_accuracies, recalls, precisions, f1_scores, losses = learner.run()

        plotting(config.mode + '_accuracy', accuracies, config.update_size)
        plotting(config.mode + '_mean_class_accuracy', mean_accuracies)
        plotting(config.mode + '_recall', recalls, config.update_size)
        plotting(config.mode + '_precision', precisions, config.update_size)
        plotting(config.mode + '_f1-score', f1_scores, config.update_size)
        plotting(config.mode + '_loss', losses, config.update_size)

        log(config, '---------- End ----------\n\n')

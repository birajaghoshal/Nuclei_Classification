import os
import config
import numpy as np
from model import Model
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
from data_handler import DataHandler
from random_learner import RandomLearner
from bootstrap_learner import BootStrap_Learner
from supervised_learner import SupervisedLearner
from uncertainty_learner import UncertaintyLearner


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
    plt.xlabel("Patches")
    plt.ylabel(title)
    plt.title(title)
    plt.savefig(config.plot_dir + title + ".png")
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == "__main__":
    config = config.load_configs()
    model = Model(config)
    data_handler = DataHandler(config)

    if os.path.isfile(config.model_path):
        os.remove(config.model_path)

    if config.mode.lower() == "supervised":
        learner = SupervisedLearner(data_handler, model, config)
        learner.run()

    elif config.mode.lower() in ["random", "uncertainty", "bootstrap"]:

        if config.auto_init:
            auto = Autoencoder(config)
            auto.train(data_handler)

        data_handler.set_training_data(np.random.choice(list(range(len(data_handler.data_x) // config.cell_patches)),
                                                        config.first_update, replace=False))
        if config.mode.lower() == "random":
            learner = RandomLearner(data_handler, model, config)
        elif config.mode.lower() == "uncertainty":
            learner = UncertaintyLearner(data_handler, model, config)
        elif config.mode.lower() == "bootstrap":
            learner = BootStrap_Learner(data_handler, model, config)

        accuracies, mean_accuracies, recalls, precisions, f1_scores, losses = learner.run()

    log(config, "---------- End ----------\n\n")

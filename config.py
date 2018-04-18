import sys
from argparse import ArgumentParser
from configparser import ConfigParser


def load_configs():
    """ Loads the arguments from the command line and config file.
    :return:
    """

    # Sets the description of the program and creates an ArgumentParser to read command line arguments.
    parser = ArgumentParser(description="Tensorflow implimentation of Active Learning for nuceli classification")

    # Creates a ConfigParser to read config file arguments.
    config = ConfigParser()

    # Loads either a given config file or default config file.
    if len(sys.argv) > 1:
        if sys.argv[1] == "--config_file":
            config.read(sys.argv[1].split('=')[1])
        else:
            config.read("config.ini")
    else:
        config.read("config.ini")

    # Standard Parameters
    parser.add_argument("--config_file", type=str, default="config.ini",
                        help="File path to configurations file.")
    parser.add_argument("--verbose", action="store_true",
                        default=config["standard"]["verbose"].lower() == "true",
                        help="Boolean if the program should display outputs while running.")
    parser.add_argument("--mode", type=str, default=config["standard"]["mode"],
                        help="The mode to run the program. [supervised, random]")
    parser.add_argument("--log_file", type=str, default=config["standard"]["log_file"],
                        help="File path to log the program's output.")
    
    # Model Parameters
    parser.add_argument("--input_height", type=int, default=int(config["model"]["input_height"]),
                        help="Integer value of the height of the input data.")
    parser.add_argument("--input_width", type=int, default=int(config["model"]["input_width"]),
                        help="Integer value of the width of the input data.")
    parser.add_argument("--input_channels", type=int, default=int(config["model"]["input_channels"]),
                        help="Integer value of the number of channels in the input data.")
    parser.add_argument("--num_classes", type=int, default=int(config["model"]["num_classes"]),
                        help="Integer value of the number of classes in the dataset.")
    parser.add_argument("--bayesian", action="store_true", default=config["model"]["bayesian"] == "True",
                        help="Boolean if the model should use MCdropout to simulate bayesian uncertainty.")
    parser.add_argument("--model_path", type=str, default=config["model"]["model_path"],
                        help="File path where the models weights shall be saved and loaded.")

    # Optimiser Parameters
    parser.add_argument('--weighted_loss', action='store_true',
                        default=config['optimiser']['weighted_loss'].lower() == 'true',
                        help='Boolean if the loss should be weighted.')
    parser.add_argument('--learning_rate', type=float, default=float(config['optimiser']['learning_rate']),
                        help='The learning rate for the model.')
    parser.add_argument('--decay', type=float, default=float(config['optimiser']['decay']),
                        help='Learning rate decay over each update.')
    parser.add_argument('--epsilon', type=float, default=float(config['optimiser']['epsilon']),
                        help='Scale the weights learning rate.')
    parser.add_argument('--use_locking', action='store_true',
                        default=config['optimiser']['use_locking'].lower() == 'true',
                        help='If locks should be used for weight updates.')

    return parser.parse_args()

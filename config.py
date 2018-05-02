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

    # Data Parameters
    parser.add_argument("--data_dir", type=str, default=config["data"]["data_dir"],
                        help="Directory path to dataset used to train the model.")
    parser.add_argument("--val_per", type=float, default=float(config["data"]["val_per"]),
                        help="Float to represent the validation percentage.")
    parser.add_argument("--balance", action="store_true",
                        default=config["data"]["balance"].lower() == "true",
                        help="Boolean if the data should be down-sampled to balance the classes.")
    parser.add_argument("--combine", type=str, default=config["data"]["combine"],
                        help="Method on how to combine the new data to the existing data. [add or replace]")
    parser.add_argument("--cell_patches", type=int, default=config["data"]["cell_patches"],
                        help="Integer for the number of patches for each cell.")
    
    # Model Parameters
    parser.add_argument("--input_height", type=int, default=int(config["model"]["input_height"]),
                        help="Integer value of the height of the input data.")
    parser.add_argument("--input_width", type=int, default=int(config["model"]["input_width"]),
                        help="Integer value of the width of the input data.")
    parser.add_argument("--input_channels", type=int, default=int(config["model"]["input_channels"]),
                        help="Integer value of the number of channels in the input data.")
    parser.add_argument("--num_classes", type=int, default=int(config["model"]["num_classes"]),
                        help="Integer value of the number of classes in the dataset.")
    parser.add_argument("--bayesian", action="store_true",
                        default=config["model"]["bayesian"].lower() == "true",
                        help="Boolean if the model should use MCdropout to simulate bayesian uncertainty.")
    parser.add_argument("--model_path", type=str, default=config["model"]["model_path"],
                        help="File path where the models weights shall be saved and loaded.")

    # Optimiser Parameters
    parser.add_argument("--weighted_loss", action="store_true",
                        default=config["optimiser"]["weighted_loss"].lower() == "true",
                        help="Boolean if the loss should be weighted.")
    parser.add_argument("--learning_rate", type=float, default=float(config["optimiser"]["learning_rate"]),
                        help="Floating point value representing the learning rate for the model.")
    parser.add_argument("--decay", type=float, default=float(config["optimiser"]["decay"]),
                        help="Floating point value representing the learning rate decay over each update.")
    parser.add_argument("--epsilon", type=float, default=float(config["optimiser"]["epsilon"]),
                        help="Floating point value to scale the weights learning rate.")
    parser.add_argument("--use_locking", action="store_true",
                        default=config["optimiser"]["use_locking"].lower() == "true",
                        help="Boolean if locks should be used for weight updates.")

    # Converge Checking Parameters
    parser.add_argument("--training_threshold", type=float, default=float(config["converge"]["training_threshold"]),
                        help="A floating point value representing the threshold for the training process.")
    parser.add_argument("--max_epochs", type=int, default=int(config["converge"]["max_epochs"]),
                        help="Integer representing the maximum number of epochs training can run for.")
    parser.add_argument("--min_epochs", type=int, default=int(config["converge"]["min_epochs"]),
                        help="Integer representing the minimum number of epochs training can run for.")
    parser.add_argument("--batch_epochs", type=int, default=int(config["converge"]["batch_epochs"]),
                        help="Integer representing the batch size of the training losses.")

    # Training Parameters
    parser.add_argument("--batch_size", type=int, default=int(config["training"]["batch_size"]),
                        help="Integer representing the size of the batches used to train the model.")
    parser.add_argument("--intervals", type=int, default=int(config["training"]["intervals"]),
                        help="Integer for the number of epochs before logging the training process.")

    # Active Learning Parameters
    parser.add_argument("--model_tuning", action="store_true",
                        default=config["active"]["model_tuning"].lower() == "true",
                        help="Boolean if the model should be fine tuning each iteration.")
    parser.add_argument("--update_size", type=int, default=int(config["active"]["update_size"]),
                        help="Integer representing the number of items to be labelled each update.")
    parser.add_argument("--pseudo_labels", action="store_true",
                        default=config["active"]["pseudo_labels"].lower() == "true",
                        help="Boolean if the model should use pseudo labels to train with.")
    parser.add_argument("--pseudo_threshold", type=float, default=float(config["active"]["pseudo_threshold"]),
                        help="Floating point value representing the threshold for adding pseudo labels.")

    # Plotting Parameters
    parser.add_argument("--plot_dir", type=str, default=config["plotting"]["plot_dir"],
                        help="Directory Path where the plots will be stored.")

    return parser.parse_args()

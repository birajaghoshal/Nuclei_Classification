import tensorflow as tf
import tensorflow.contrib.layers as layers


class Model:
    def __init__(self, config):
        """ Initialiser for the Model class.
        :param config: A ArgumentParser object
        """

        # Resets the Tensorflow computational graph.
        tf.reset_default_graph()

        # Sets the class variables from the config file.
        self.input_shape = [config.input_height, config.input_width, config.input_channels]
        self.num_classes = config.num_classes
        self.verbose = config.verbose
        self.config = config

        # Sets the tensor placeholders
        self.X = tf.placeholder("float", [None] + self.input_shape, name='X')
        self.Y = tf.placeholder("float", [None, self.num_classes], name='Y')

        # Creates the classification model
        self.model = self.create_model()
        self.log("Classification Model has been created\n")

    def __copy__(self):
        """ Resets the model and returns a copy of the model.
        :return: A reset copy of the reset Model
        """

        tf.reset_default_graph()
        return Model(self.config)

    def log(self, message):
        """ Function to handle printing and logging of messages.
        :param message: String of message to be printed and logged.
        """

        if self.config.verbose:
            print(message)
        if self.config.log_file != '':
            print(message, file=open(self.config.log_file, 'a'))

    def create_model(self):
        """ Creates a CNN model for Classification.
        :return: A computational graph representing a CNN model for Classification.
        """

        def max_pool(input_tensor, k, d, name):
            """ Creates a Max Pooling layer for the model
            :param input_tensor: The tensor for the max pooling layer to use as input.
            :param k: The size of the filters.
            :param d: The size of the strides.
            :param name: The name of the operation.
            :return: A tensor.
            """

            return tf.nn.max_pool(input_tensor, ksize=[1, k, k, 1], strides=[1, d, d, 1], padding='VALID', name=name)

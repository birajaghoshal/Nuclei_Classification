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
            """ Creates a Max Pooling layer operation.
            :param input_tensor: The tensor for the max pooling layer to use as input.
            :param k: The size of the filters.
            :param d: The size of the strides.
            :param name: The name of the operation.
            :return: A tensor.
            """

            return tf.nn.max_pool(input_tensor, ksize=[1, k, k, 1], strides=[1, d, d, 1], padding="VALID", name=name)

        def convolution(input_tensor, k, d, n_out, name, activation_fn=tf.nn.relu, batch_norm=True):
            """ Creates a Convolutional layer operation.
            :param input_tensor: The tensor for the Convolutional layer will use as input.
            :param k: The size of the filters.
            :param d: The size of the strides.
            :param n_out: The number of output filters.
            :param name: The name of the operation
            :param activation_fn: The activation function to be applied.
            :param batch_norm: If batch normalisation should be applied.
            :return: A tensor.
            """

            # Get the number of filters from the input tensor.
            n_in = input_tensor.get_shape()[-1].value

            with tf.variable_scope(name):
                # Declares the weghts and biases for the convolutional layer.
                weights = tf.get_variable("weights", [k, k, n_in, n_out], tf.float32, layers.xavier_initializer())
                biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0))

                # Creates the convolutional tensor operations with the declared weights and biases.
                conv = tf.nn.conv2d(input_tensor, weights, (1, d, d, 1), padding="VALID")
                logits = tf.nn.bias_add(conv, biases)

                # Creates the operation for the activation applied to the logits.
                activations = activation_fn(logits)

                # Creates the batch normalisation operation if required.
                if batch_norm:
                    return layers.batch_norm(activations)
                else:
                    return activations

        def fully_connected(input_tensor, n_out, name, activation_fn=tf.nn.relu, batch_norm=True):
            """ Creates a fully connected layer operation.
            :param input_tensor: The tensor the fully connected layer use take as input.
            :param n_out: The number of output filters.
            :separam name: THe name of the operation.
            :param activation_fn: The activation function to be applied.
            :param batch_norm:If batch normalisation should be applied.
            :return: A tensor.
            """

            # Get the number of filters from the input tensor.
            n_in = input_tensor.get_shape()[-1].value

            with tf.variable_scope(name):
                # Declares the weights and biases for the fully connected layer.
                weights = tf.get_variable("weights", [n_in, n_out], tf.float32, layers.xavier_initializer())
                biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0))

                # Creates the fully connected operation with the declared weights and biases.
                logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)

                # Creates the operation for the activation applied to the logits.
                activations = activation_fn(logits)

                # Creates the batch normalisation operation if required.
                if batch_norm:
                    return layers.batch_norm(activations)
                else:
                    return activations

        def linear(input):
            """ Linear activation function.
            :param input: Input Tensor.
            :return: The same Input tensor.
            """

            return input

        # Block 1
        model = convolution(self.X, 3, 1, 64, "Conv_1.1")
        model = convolution(model, 3, 1, 64, "Conv_1.2")
        model = max_pool(model, 2, 2, "Pool_1.1")

        # Block 2
        model = convolution(model, 3, 1, 128, "Conv_2.1")
        model = convolution(model, 3, 1, 128, "Conv_2.2")
        model = max_pool(model, 2, 2, "Pool_2.1")

        # Block 3
        model = convolution(model, 3, 1, 256, "Conv_3.1")
        model = convolution(model, 3, 1, 256, "Conv_3.2")
        model = convolution(model, 3, 1, 256, "Conv_3.3")
        model = max_pool(model, 2, 2, "Pool_3.1")

        # Block 4
        model = tf.contrib.layers.flatten(model)
        model = fully_connected(model, 1024, "Full_1")
        model = fully_connected(model, 1024, "Full_2")

        # Output
        return fully_connected(model, self.num_classes, "Output", activation_fn=linear)

    def optimiser(self, weights):
        """ Creates the loss and optimiser operations for training the model.
        :param weights: A set of weights based on the class count to be applied to the logits.
        :return: Two tensor operations to calculate loss and to optimise based on loss
        """

        # Creates the loss operation with the output of the model and the labels.
        if self.config.weighted_loss:
            # Creates a tensor with the weights.
            weights = tf.constant(weights, dtype=tf.float32)

            # Multiplies the logits with the weights.
            logits = tf.multiply(self.model, weights)

            # Creates the loss operation with the weights logits.
            loss = tf.losses.softmax_cross_entropy(onehot_labels=self.Y, logits=logits)
        else:
            # Creates the loss operation.
            loss = tf.losses.softmax_cross_entropy(onehot_labels=self.Y, logits=self.model)

        # Creates the optimiser with the parameters from the configs.
        optimiser = tf.train.AdadeltaOptimizer(self.config.learning_rate, self.config.decay,
                                               self.config.epsilon, self.config.use_locking)

        # Creates the optimiser operation based on the loss.
        optimiser = optimiser.minimize(loss)

        # Returns both the loss and optimiser operations.
        return loss, optimiser


import types
import numpy as np
import tensorflow as tf

class Neural_Network():
    """Base class for various neural networks (consider using
       abstract base classes with 'from abc import ABCMeta')"""


    def __init__(self, architecture, transfer_funcs):

        # The below is hardly error proof. It is just a few minor checks to help
        # with debugging. Eventually it will need to be bolstered further (check
        # architecture too)
        if hasattr(transfer_funcs, '__iter__'): # Check to see if it is iterable
            # If it is iterable that means they put in a list or tuple (hopefully)
            # in which case they must specify an actiavation func for each layer
            ERR_MSG = 'Must specify a transfer function for each layer. i.e. \
            len(architecture) must equal len(transfer_func)'
            assert len(architecture) == len(transfer_funcs), ERR_MSG
        elif isinstance(transfer_funcs, types.FunctionType): # See if it's a func
            # Then duplicate it for every layer. If the user only puts in one
            # instance, it means they want it for every layer
            transfer_funcs = [transfer_funcs]*len(architecture)

        self.architecture = architecture
        self.transfer_funcs = transfer_funcs
        self.weights = []
        self.biases = []


    def xavier_init(self, input_size, output_size):
        size = np.sqrt(6.0/(input_size + output_size))
        return tf.random_uniform((input_size, output_size),
                                 minval=-size, maxval=size,
                                 dtype=tf.float32)


class DNN(Neural_Network):
    """Deep Neural Network - Standard Multilayer Perceptron model (MLP)"""


    def __init__(self, architecture, transfer_funcs):
        Neural_Network.__init__(self, architecture, transfer_funcs)


    def build_graph(self, network_input, input_dim):

        num_prev_nodes = input_dim  # Number of nodes in the input
        # Currently I am not keeping track of the output between layers
        current_input = network_input
        for func, num_next_nodes in zip(self.transfer_funcs, self.architecture):
            init_weight_val = self.xavier_init(num_prev_nodes, num_next_nodes)
            weight = tf.Variable(initial_value=init_weight_val, dtype=tf.float32)
            self.weights.append(weight)
            init_bias_val = np.zeros((1,num_next_nodes))
            bias = tf.Variable(initial_value=init_bias_val, dtype=tf.float32)
            self.biases.append(bias)
            num_prev_nodes = num_next_nodes
            current_input = func(current_input @ weight + bias)

        return current_input

    def get_output_dim(self):

        return self.architecture[-1]


class CNN(Neural_Network):

    def __init__(self):
        None

    def build_graph(self):
        None


class DCNN(Neural_Network):

    def __init__(self):
        None

    def build_graph(self):
        None


class CNN3D(Neural_Network):

    def __init__(self):
        None

    def build_graph(self):
        None


class DCNN3D(Neural_Network):

    def __init__(self):
        None

    def build_graph(self):
        None

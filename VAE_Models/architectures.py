import types
import numpy as np
import tensorflow as tf
from IPython import embed
import sys

class Neural_Network():
    """Base class for various neural networks (consider using
       abstract base classes with 'from abc import ABCMeta')"""


    def __init__(self, architecture):

        self.architecture = architecture
        self.weights = []
        self.biases = []


    def xavier_init(self, shape):
        with tf.name_scope('Xavier_Init'):
            input_size = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
            output_size = shape[1] if len(shape) == 2 else shape[-1]
            size = np.sqrt(6.0/(input_size + output_size))
            return tf.random_uniform(shape, minval=-size,
                                     maxval=size,
                                     dtype=tf.float32)


class DNN(Neural_Network):
    """Deep Neural Network - Standard Multilayer Perceptron model (MLP)"""


    def __init__(self, architecture, transfer_funcs):
        Neural_Network.__init__(self, architecture)

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

        self.transfer_funcs = transfer_funcs

    def build_graph(self, network_input, input_shape, scope='DNN'):

        with tf.name_scope(scope):
            num_prev_nodes = np.prod(input_shape)
            # Currently I am not keeping track of the output between layers
            current_input = network_input
            for func, num_next_nodes in zip(self.transfer_funcs, self.architecture):
                init_weight_val = self.xavier_init((num_prev_nodes, num_next_nodes))
                weight = tf.Variable(initial_value=init_weight_val,
                        dtype=tf.float32, name='Weight')
                self.weights.append(weight)
                init_bias_val = np.zeros((1,num_next_nodes))
                bias = tf.Variable(initial_value=init_bias_val,
                        dtype=tf.float32, name='Bias')
                self.biases.append(bias)
                num_prev_nodes = num_next_nodes
                current_input = func(current_input @ weight + bias)

        return current_input

    def get_output_dim(self):

        return self.architecture[-1]


class CNN(Neural_Network):

    def __init__(self, architecture):
        Neural_Network.__init__(self, architecture)
        self.channels = architecture['channels']
        self.filterSize = architecture['filterSize']
        self.outputShape = architecture['outputShape']
        self.convolution_counter = 1
        #self.strides = architecture['strides']

    def add_layer(self, layer_input, filterSz, prevChannels, nextChannels):
        cc = self.convolution_counter # Which convolution is this?
        with tf.name_scope("Convolution_%d" % (cc)):
            convWeightshape = [filterSz,filterSz,prevChannels,nextChannels]
            convWeights = tf.Variable(self.xavier_init(convWeightshape),
                    name='conv_weights%d' % (cc))
            bias = tf.Variable(tf.zeros([nextChannels]), name='bias%d' % (cc))

            # Convolutional Layer #1
            conv = tf.nn.conv2d(layer_input, convWeights,
                                    strides=[1, 1, 1, 1],
                                    padding="SAME",
                                    name='conv%d' % (cc)) # Pad the input so result of convolution is same size, i.e. (28,28)

            conv_output = tf.nn.elu(tf.add(conv, bias), name='conv%d_output' % (cc))


        self.convolution_counter += 1
        return conv_output


    def build_graph(self, network_input, input_shape, scope='CNN'):
        """The documentation for the build_graph routine of the CNN class. To
        come..."""

        img_h, img_w = input_shape
        current_input = tf.reshape(network_input, (-1,img_h, img_w,1))
        prevChannels = 1
        with tf.name_scope(scope):
            for filterSz,numChannels in zip(self.filterSize,self.channels):
                current_input = self.add_layer(current_input, filterSz, prevChannels, numChannels)
                prevChannels = numChannels

        fcWeights = tf.Variable(self.xavier_init([img_h, img_w,
            prevChannels, self.outputShape]), dtype=tf.float32,
            name='fc_weights')
        fcBias = tf.Variable(tf.zeros([self.outputShape]),
                name='fc_bias')

        fcOutput = tf.nn.elu(tf.add(tf.tensordot(current_input, fcWeights,
            [[1,2,3], [0,1,2]]), fcBias), name='fc_output')


        return fcOutput


    def get_output_dim(self):

        return self.outputShape


class DCNN(Neural_Network):

    def __init__(self):
        None

    def build_graph(self):
         """
            self.meanWeights = tf.Variable(self.xavier_init([img_h,img_w,channels,self.hiddenNodes]), name='meanWeights')
            self.sigmaWeights = tf.Variable(self.xavier_init([img_h,img_w,channels,self.hiddenNodes]), name='sigmaWeights')
            self.meanBias = tf.Variable(tf.zeros([self.hiddenNodes]), name='meanBias')
            self.sigmaBias = tf.Variable(tf.zeros([self.hiddenNodes]), name='sigmaBias')

            # Lantecy Space
            self.z_mean = tf.add(tf.tensordot(self.conv_output, self.meanWeights, [[1,2,3],[0,1,2]]), self.meanBias)
            self.z_log_sigma_sq = tf.add(tf.tensordot(self.conv_output, self.sigmaWeights, [[1,2,3],[0,1,2]]), self.sigmaBias)

            epsilons = tf.random_normal(tf.shape(self.z_log_sigma_sq), name="epsilon")
            self.z_sample = tf.add(self.z_mean, tf.multiply(self.z_log_sigma_sq, epsilons))
            self.denseWeights2 =
            tf.Variable(self.xavier_init([self.hiddenNodes,img_h,img_w,channels]), name='weights2')
            self.denseBias2 = tf.Variable(tf.zeros([channels]), name='bias2')

            self.deconv_input = tf.nn.relu(tf.add(tf.tensordot(self.z_sample,
                self.denseWeights2, [[1],[0]]), self.denseBias2), name='deconv_input')

            # Deconvolutional Layer #1
            self.deconv = tf.nn.conv2d_transpose(self.deconv_input, self.convWeights,
                                    #output_shape=tf.shape(self.x_input),
                                    output_shape=[self.batch_size, self.winSize, self.winSize, 1],
                                    strides=[1, 1, 1, 1],
                                    padding="SAME",
                                    name='deconv') # Pad the input so result of convolution is same size, i.e. (28,28)

            self.convBias2 = tf.Variable(tf.zeros([self.inputSize]), name='convBias2')
            self.deconv2 = tf.reshape(self.deconv, [self.batch_size,self.winSize*self.winSize])
            self.x_reconstruction = tf.sigmoid(tf.add(self.deconv2, self.convBias2))
        """


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

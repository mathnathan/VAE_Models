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
        """filterSizes should be a list of tuples...
            [(filter1_height, filter1_width), (filter2_height, filter2_width), ... ]

            channels should be a list of integers...
            [8,16,13,...]

            len(filterSizes) must equal len(channels)
        """
        Neural_Network.__init__(self, architecture)
        self.channels = architecture['channels']
        self.filterSizes = architecture['filterSizes']
        self.fc_layer_size = architecture['fc_layer_size']
        self.convolution_counter = 1
        assert len(self.filterSizes) == len(self.channels)


    def add_layer(self, layer_input, filterSz, prevChannels, nextChannels):
        cc = self.convolution_counter # Which convolution is this?
        with tf.name_scope("Convolution_%d" % (cc)):
            convWeightshape = [filterSz[0],filterSz[1],prevChannels,nextChannels]
            convWeights = tf.Variable(self.xavier_init(convWeightshape),
                    name='conv_weights%d' % (cc))
            bias = tf.Variable(tf.zeros([nextChannels]), name='bias%d' % (cc))

            # Convolutional Layer #1
            conv = tf.nn.conv2d(layer_input, convWeights,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
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
            for filterSz,numChannels in zip(self.filterSizes,self.channels):
                current_input = self.add_layer(current_input, filterSz, prevChannels, numChannels)
                prevChannels = numChannels

        fcWeights = tf.Variable(self.xavier_init([img_h, img_w,
            prevChannels, self.fc_layer_size]), dtype=tf.float32,
            name='fc_weights')
        fcBias = tf.Variable(tf.zeros([self.fc_layer_size]),
                name='fc_bias')

        fcOutput = tf.nn.elu(tf.add(tf.tensordot(current_input, fcWeights,
            [[1,2,3], [0,1,2]]), fcBias), name='fc_output')


        return fcOutput


    def get_output_dim(self):

        return self.fc_layer_size


class CNN3D(Neural_Network):


    def __init__(self, architecture):
        """ filterSizes should be a list of tuples...
            [(f1_depth, f1_height, f1_width), (f2_depth, f2_height, f2_width), ... ]

            channels should be a list of integers...
            [8,16,13,...]

            len(filterSizes) must equal len(channels)
        """

        Neural_Network.__init__(self, architecture)
        self.channels = architecture['channels']
        self.filterSizes = architecture['filterSizes']
        self.fc_layer_size = architecture['fc_layer_size']
        self.convolution_counter = 1
        assert len(self.filterSizes) == len(self.channels)


    def add_layer(self, layer_input, filterSz, prevChannels, nextChannels):
        cc = self.convolution_counter # Which convolution is this?
        with tf.name_scope("3DConvolution_%d" % (cc)):
            convWeightshape = [filterSz[0],filterSz[1],filterSz[2],prevChannels,nextChannels]
            convWeights = tf.Variable(self.xavier_init(convWeightshape),
                    name='3Dconv_weights%d' % (cc))
            bias = tf.Variable(tf.zeros([nextChannels]), name='bias%d' % (cc))

            # Convolutional Layer #1
            conv = tf.nn.conv3d(layer_input, convWeights,
                                    strides=[1, 1, 1, 1, 1],
                                    padding="SAME",
                                    name='3Dconv%d' % (cc)) # Pad the input so result of convolution is same size, i.e. (28,28)

            conv_output = tf.nn.elu(tf.add(conv, bias), name='3Dconv%d_output' % (cc))


        self.convolution_counter += 1
        return conv_output


    def build_graph(self, network_input, input_shape, scope='3DCNN'):
        """The documentation for the build_graph routine of the CNN class. To
        come..."""

        img_d, img_h, img_w = input_shape
        current_input = tf.reshape(network_input, (-1, img_d, img_h, img_w, 1))
        prevChannels = 1
        with tf.name_scope(scope):
            for filterSz, numChannels in zip(self.filterSizes,self.channels):
                current_input = self.add_layer(current_input, filterSz, prevChannels, numChannels)
                prevChannels = numChannels

        fcWeights = tf.Variable(self.xavier_init([img_d, img_h, img_w,
            prevChannels, self.fc_layer_size]), dtype=tf.float32,
            name='fc_weights')
        fcBias = tf.Variable(tf.zeros([self.fc_layer_size]),
                name='fc_bias')

        fcOutput = tf.nn.elu(tf.add(tf.tensordot(current_input, fcWeights,
            [[1,2,3,4], [0,1,2,3]]), fcBias), name='fc_output')


        return fcOutput


    def get_output_dim(self):

        return self.fc_layer_size


class DCNN(Neural_Network):


    def __init__(self, architecture, batch_size):
        Neural_Network.__init__(self, architecture)
        self.channels = architecture['channels']
        self.outputShape = architecture['outputShape']
        self.batch_size = batch_size
        self.final_output_shape = self.outputShape[0]*self.outputShape[1]
        self.deconvolution_counter = 1


    def add_layer(self, layer_input, input_height, input_width, prevChannels,
            nextChannels, outputShape):
        print("outputShape = ", outputShape)
        dc = self.deconvolution_counter # Which convolution is this?
        with tf.name_scope("Deconvolution_%d" % (dc)):
            deconvWeightshape = [input_height,input_width,nextChannels,prevChannels]
            deconvWeights = tf.Variable(self.xavier_init(deconvWeightshape),
                    name='deconv_weights%d' % (dc))
            bias = tf.Variable(tf.zeros([nextChannels]), name='bias%d' % (dc))

            out = [self.batch_size, outputShape[0], outputShape[1], nextChannels]
            deconv = tf.nn.conv2d_transpose(layer_input, deconvWeights,
                                    out,
                                    strides=[1, 1, 1, 1],
                                    padding="SAME",
                                    name='deconv%d' % (dc)) # Pad the input so result of convolution is same size, i.e. (28,28)

            deconv_output = tf.nn.elu(tf.add(deconv, bias), name='deconv%d_output' % (dc))


        self.deconvolution_counter += 1
        return deconv_output


    def build_graph(self, network_input, input_shape, scope='CNN'):
        """The documentation for the build_graph routine of the CNN class. To
        come..."""

        current_input = tf.reshape(network_input, (-1,input_shape,1,1))
        prevChannels = 1
        input_height = input_shape
        input_width = 1
        with tf.name_scope(scope):
            for numChannels, outputShape in zip(self.channels,self.outputShapes):
                current_input = self.add_layer(current_input, input_height,
                        input_width, prevChannels, numChannels, outputShape)
                input_height, input_width = outputShape
                prevChannels = numChannels

        deconvWeightshape = [input_height,input_width,1,prevChannels]
        deconvWeights = tf.Variable(self.xavier_init(deconvWeightshape),
                name='final_deconv_weights')
        bias = tf.Variable(tf.zeros([1]), name='final_bias')

        out = [self.batch_size, outputShape[0], outputShape[1], 1]
        deconv = tf.nn.conv2d_transpose(current_input, deconvWeights,
                                out,
                                strides=[1, 1, 1, 1],
                                padding="SAME",
                                name='final_deconv') # Pad the input so result of convolution is same size, i.e. (28,28)

        final_deconv_output = tf.nn.elu(tf.add(deconv, bias), name='final_deconv_output')


        return tf.reshape(final_deconv_output, (-1,self.final_output_shape))


    def get_output_dim(self):

        return self.final_output_shape

class DCNN3D(Neural_Network):


    def __init__(self):
        None


    def build_graph(self):
        None

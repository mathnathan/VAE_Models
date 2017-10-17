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
                                     dtype=self.dtype)


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


    def build_graph(self, network_input, input_shape, dtype=tf.float32, scope='DNN'):

        with tf.name_scope(scope):
            self.dtype = dtype
            num_prev_nodes = np.prod(input_shape)
            # Currently I am not keeping track of the output between layers
            current_input = network_input
            for func, num_next_nodes in zip(self.transfer_funcs, self.architecture):
                init_weight_val = self.xavier_init((num_prev_nodes, num_next_nodes))
                weight = tf.Variable(initial_value=init_weight_val,
                        dtype=self.dtype, name='Weight')
                self.weights.append(weight)
                init_bias_val = np.zeros((1,num_next_nodes))
                bias = tf.Variable(initial_value=init_bias_val,
                        dtype=self.dtype, name='Bias')
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
            [8,16,13, ...]

            strides should be a list of arrays of 4 elements...
            [[1,2,2,1], [1,1,1,1], ...]

            len(filterSizes) must equal len(channels)
        """
        Neural_Network.__init__(self, architecture)
        self.channels = architecture['channels']
        self.filterSizes = architecture['filterSizes']
        self.strides = architecture['strides']
        self.padding = architecture['padding']
        self.fc_layer_size = architecture['fc_layer_size']
        self.convolution_counter = 1
        assert len(self.filterSizes) == len(self.channels)


    def add_layer(self, layer_input, filterSz, prevChannels,
                        nextChannels, outputShape, strides):
        cc = self.convolution_counter # Which convolution is this?
        with tf.name_scope("Convolution_%d" % (cc)):
            convWeightshape = [filterSz[0],filterSz[1],prevChannels,nextChannels]
            convWeights = tf.Variable(self.xavier_init(convWeightshape), dtype=self.dtype,
                    name='conv_weights%d' % (cc))
            bias = tf.Variable(tf.zeros(outputShape, dtype=self.dtype),
                    name='bias%d' % (cc))

            # Convolutional Layer #1
            conv = tf.nn.conv2d(layer_input, convWeights,
                                    strides=strides,
                                    padding=self.padding,
                                    name='conv%d' % (cc))

            conv_output = tf.nn.elu(tf.add(conv, bias), name='conv%d_output' % (cc))

        self.convolution_counter += 1
        return conv_output


    def build_graph(self, network_input, input_shape, dtype=tf.float32, scope='CNN'):
        """The documentation for the build_graph routine of the CNN class. To
        come..."""

        self._calculate_output_sizes(network_input, input_shape)
        self.dtype = dtype
        self.input_shape = input_shape
        img_h, img_w, img_c = self.input_shape
        current_input = tf.reshape(network_input, (-1, img_h, img_w, img_c))
        pc = img_c
        with tf.name_scope(scope):
            for f,nc,sh,st in zip(self.filterSizes, self.channels,
                    self.output_shapes, self.strides):
                current_input = self.add_layer(current_input, f, pc, nc, sh, st)
                pc = nc

        output_shape = self.output_shapes[-1]
        fcWeights = tf.Variable(self.xavier_init([output_shape[0], output_shape[1],
            nc, self.fc_layer_size]), dtype=self.dtype,
            name='fc_weights')
        fcBias = tf.Variable(tf.zeros([self.fc_layer_size], dtype=self.dtype),
                name='fc_bias')

        fcOutput = tf.nn.elu(tf.add(tf.tensordot(current_input, fcWeights,
            [[1,2,3], [0,1,2]]), fcBias), name='fc_output')

        return fcOutput


    def _calculate_output_sizes(self, network_input, input_shape):

        self.output_shapes = []
        img_h, img_w, img_c = input_shape
        prev_h = img_h
        prev_w = img_w
        if self.padding == "SAME":
            for c in channels:
                self.output_shapes.append((img_h,img_w,img_c))
        elif self.padding == "VALID":
            for f,c,s in zip(self.filterSizes, self.channels, self.strides):
                height = (prev_h - f[0]) / s[1] + 1
                width = (prev_w - f[1]) / s[2] + 1
                self.output_shapes.append((height, width, c))
                prev_h = height
                prev_w = width
        else:
            raise KeyError('padding must be either "SAME" or "VALID"')

        self.output_shapes = np.array(self.output_shapes, dtype=np.int16)


    def get_output_dim(self):

        return self.fc_layer_size


    def print_network_details(self):

        print("\ndtype = ", self.dtype)
        print("\ninput_shape = ", self.input_shape)
        print("\nchannels = ", self.channels)
        print("\nfilter_sizes = ", self.filterSizes)
        print("\nstrides = ", self.strides)
        print("\npadding = ", self.padding)
        print("\noutput_sizes = ", self.output_shapes)
        print("\nfc_layer_size = ", self.fc_layer_size)


class DCNN(Neural_Network):


    def __init__(self, architecture, batch_size):
        Neural_Network.__init__(self, architecture)
        self.channels = architecture['channels']
        self.outputShape = architecture['outputShape']
        self.filterSizes = architecture['filterSizes']
        self.padding = architecture['padding']
        self.strides = architecture['strides']
        self.batch_size = batch_size
        self.final_output_shape = self.outputShape[0]*self.outputShape[1]
        self.deconvolution_counter = 1


    def add_layer(self, layer_input, filterSz, prevChannels,
                    nextChannels, outputShape):
        dc = self.deconvolution_counter # Which deconvolution is this?
        with tf.name_scope("Deconvolution_%d" % (dc)):
            deconvWeightshape = [filterSz[0], filterSz[1],
                    nextChannels, prevChannels]
            deconvWeights = tf.Variable(self.xavier_init(deconvWeightshape),
                    dtype=self.dtype, name='deconv_weights%d' % (dc))
            bias = tf.Variable(tf.zeros([nextChannels], dtype=self.dtype),
                    name='bias%d' % (dc))
            out = [self.batch_size, outputShape[0], outputShape[1],
                    nextChannels]
            deconv = tf.nn.conv2d_transpose(layer_input, deconvWeights,
                                        out,
                                        strides=[1, 1, 1, 1],
                                        padding=self.padding,
                                        name='deconv%d' % (dc))

            deconv_output = tf.nn.elu(tf.add(deconv, bias),
                    name='deconv%d_output' % (dc))


        self.deconvolution_counter += 1
        return deconv_output


    def build_graph(self, network_input, input_shape, dtype=tf.float32,
            scope='CNN'):
        """The documentation for the build_graph routine of the DCNN class. To
        come..."""

        self._calculate_output_sizes(self, network_input, input_shape):
        self.dtype = dtype
        img_h, img_w, img_c = self.outputShape
        # First we take the 1D latency space and pass through a FC layer
        # that will be reshaped into a 2D image to be deconvolved.
        fcWeights = tf.Variable(self.xavier_init([input_shape, img_h, img_w]),
                                dtype=self.dtype, name='fc_weights')
        fcBias = tf.Variable(tf.zeros([img_h, img_w], dtype=self.dtype),
                name='fc_bias')

        fcOutput = tf.nn.elu(tf.add(tf.tensordot(network_input, fcWeights,
            [[1], [0]]), fcBias), name='fc_output')

        current_input = tf.expand_dims(fcOutput, -1)
        prevChannels = img_c
        with tf.name_scope(scope):
            for filterSz,numChannels in zip(self.filterSizes,self.channels):
                current_input = self.add_layer(current_input, filterSz,
                                        prevChannels, numChannels, self.outputShape)
                prevChannels = numChannels


        deconvWeightshape = [input_height,input_width,1,prevChannels]
        deconvWeights = tf.Variable(self.xavier_init(deconvWeightshape),
                dtype=self.dtype, name='final_deconv_weights')
        bias = tf.Variable(tf.zeros([1], dtype=self.dtype), name='final_bias')

        out = [self.batch_size, outputShape[0], outputShape[1], 1]
        deconv = tf.nn.conv2d_transpose(current_input, deconvWeights,
                                out,
                                strides=[1, 1, 1, 1],
                                padding="SAME",
                                name='final_deconv') # Pad the input so result of convolution is same size, i.e. (28,28)

        final_deconv_output = tf.nn.sigmoid(tf.add(deconv, bias), name='final_deconv_output')


        return tf.reshape(final_deconv_output, (-1,self.final_output_shape))


    def _calculate_output_sizes(self, network_input, input_shape):

        self.output_shapes = []
        img_h, img_w, img_c = input_shape
        prev_h = img_h
        prev_w = img_w
        if self.padding == "SAME":
            for c in channels:
                self.output_shapes.append((img_h,img_w,img_c))
        elif self.padding == "VALID":
            for f,c,s in zip(self.filterSizes, self.channels, self.strides):
                height = (prev_h - f[0]) / s[1] + 1
                width = (prev_w - f[1]) / s[2] + 1
                self.output_shapes.append((height, width, c))
                prev_h = height
                prev_w = width
        else:
            raise KeyError('padding must be either "SAME" or "VALID"')

        self.output_shapes = np.array(self.output_shapes, dtype=np.int16)


    def get_output_dim(self):

        return self.final_output_shape


    def print_network_details(self):

        print("\ndtype = ", self.dtype)
        print("\ninput_shape = ", self.input_shape)
        print("\nchannels = ", self.channels)
        print("\nfilter_sizes = ", self.filterSizes)
        print("\nstrides = ", self.strides)
        print("\npadding = ", self.padding)
        print("\noutput_sizes = ", self.output_shapes)
        print("\nfc_layer_size = ", self.fc_layer_size)


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
            convWeights = tf.Variable(self.xavier_init(convWeightshape), dtype=self.dtype,
                    name='3Dconv_weights%d' % (cc))
            bias = tf.Variable(tf.zeros([nextChannels], dtype=self.dtype),
                    name='bias%d' % (cc))

            # Convolutional Layer #1
            conv = tf.nn.conv3d(layer_input, convWeights,
                                    strides=[1, 1, 1, 1, 1],
                                    padding="SAME",
                                    name='3Dconv%d' % (cc)) # Pad the input so result of convolution is same size, i.e. (28,28)

            conv_output = tf.nn.elu(tf.add(conv, bias), name='3Dconv%d_output' % (cc))


        self.convolution_counter += 1
        return conv_output


    def build_graph(self, network_input, input_shape, dtype=tf.float32, scope='3DCNN'):
        """The documentation for the build_graph routine of the CNN class. To
        come..."""

        self.dtype = dtype
        img_d, img_h, img_w, img_c = input_shape
        current_input = tf.reshape(network_input, (-1, img_d, img_h, img_w, img_c))
        prevChannels = img_c
        with tf.name_scope(scope):
            for filterSz, numChannels in zip(self.filterSizes,self.channels):
                current_input = self.add_layer(current_input, filterSz, prevChannels, numChannels)
                prevChannels = numChannels

        fcWeights = tf.Variable(self.xavier_init([img_d, img_h, img_w,
            prevChannels, self.fc_layer_size]), dtype=self.dtype, name='fc_weights')
        fcBias = tf.Variable(tf.zeros([self.fc_layer_size], dtype=self.dtype),
                name='fc_bias')

        fcOutput = tf.nn.elu(tf.add(tf.tensordot(current_input, fcWeights,
            [[1,2,3,4], [0,1,2,3]]), fcBias), name='fc_output')


        return fcOutput


    def get_output_dim(self):

        return self.fc_layer_size



class DCNN3D(Neural_Network):


    def __init__(self):
        None


    def build_graph(self):
        None

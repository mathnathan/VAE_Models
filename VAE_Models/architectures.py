import types
import numpy as np
import tensorflow as tf
from IPython import embed
import sys

# TODO: Add replace tf.Variable with tf.get_variable


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
            print("input_size = ", input_size)
            output_size = shape[1] if len(shape) == 2 else shape[-1]
            print("output_size = ", output_size)
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


    def _build_graph(self, network_input, input_shape, dtype=tf.float32, scope='DNN'):

        with tf.name_scope(scope):
            self.dtype = dtype
            num_prev_nodes = input_shape if isinstance(input_shape, int) else np.prod(input_shape[:-1])
            # Currently I am not keeping track of the output between layers
            current_input = network_input
            for func, num_next_nodes in zip(self.transfer_funcs, self.architecture):
                print("num_prev_nodes = ", num_prev_nodes)
                print("num_next_nodes = ", num_next_nodes)
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


    def get_output_shape(self):

        output_layer = self.architecture[-1]
        return output_layer


class CNN(Neural_Network):


    def __init__(self, architecture):
        """filter_sizes should be a list of tuples...
            [(filter1_height, filter1_width), (filter2_height, filter2_width), ... ]

            channels should be a list of integers...
            [8,16,13, ...]

            strides should be a list of arrays of 4 elements...
            [[1,2,2,1], [1,1,1,1], ...]

            len(filter_sizes) must equal len(channels)
        """
        Neural_Network.__init__(self, architecture)
        self.channels = architecture['channels']
        self.filter_sizes = architecture['filter_sizes']
        self.strides = architecture['strides']
        self.fc_layer_size = architecture['fc_layer_size']
        self.convolution_counter = 1
        assert len(self.filter_sizes) == len(self.channels)


    def add_layer(self, layer_input, output_shape, filter_sz, prev_channels,
                        channels, strides):
        cc = self.convolution_counter # Which convolution is this?
        with tf.name_scope("Convolution_%d" % (cc)):
            convWeightshape = [filter_sz[0],filter_sz[1],prev_channels,channels]
            convWeights = tf.Variable(self.xavier_init(convWeightshape), dtype=self.dtype,
                    name='conv_weights%d' % (cc))
            bias = tf.Variable(tf.zeros(output_shape, dtype=self.dtype),
                    name='bias%d' % (cc))

            # Convolutional Layer #1
            conv = tf.nn.conv2d(layer_input, convWeights,
                                    strides=strides,
                                    padding='SAME',
                                    name='conv%d' % (cc))

            conv_output = tf.nn.relu(tf.add(conv, bias), name='conv%d_output' % (cc))

        self.convolution_counter += 1
        return conv_output


    def _build_graph(self, network_input, input_shape, dtype=tf.float32, scope='CNN'):
        """The documentation for the build_graph routine of the CNN class. To
        come..."""

        ph, pw, c = input_shape
        self.dtype = dtype
        current_input = tf.reshape(network_input, (-1, ph, pw, c))
        pc = c
        print("Convolution")
        with tf.name_scope(scope):
            for f,c,s in zip(self.filter_sizes, self.channels, self.strides):
                print("input_shape = ", input_shape)
                print("filter = ", f)
                print("channels = ", c)
                print("strides = ", s)
                h = ph // s[1]
                w = pw // s[2]
                out_shape = (h, w, c)
                print("output_shape = ", out_shape)
                current_input = self.add_layer(current_input, out_shape, f, pc, c, s)
                print("current_input.shape = ", current_input.shape)
                pc = c # previous channels become the current output channels
                ph = h
                pw = w

        fcWeights = tf.Variable(self.xavier_init([out_shape[0], out_shape[1],
            out_shape[2], self.fc_layer_size]), dtype=self.dtype,
            name='fc_weights')
        fcBias = tf.Variable(tf.zeros([self.fc_layer_size], dtype=self.dtype),
                name='fc_bias')

        fcOutput = tf.nn.relu(tf.add(tf.tensordot(current_input, fcWeights,
            [[1,2,3], [0,1,2]]), fcBias), name='fc_output')

        return fcOutput


    def print_network_details(self):

        print("\ndtype = ", self.dtype)
        print("\ninput_shape = ", self.input_shape)
        print("\nchannels = ", self.channels)
        print("\nfilter_sizes = ", self.filter_sizes)
        print("\nstrides = ", self.strides)
        print("\noutput_sizes = ", self.output_shapes)
        print("\nfc_layer_size = ", self.fc_layer_size)


    def get_output_shape(self):

        return self.fc_layer_size


class UPCNN(Neural_Network):


    def __init__(self, architecture):

        Neural_Network.__init__(self, architecture)

        self.filter_sizes = architecture['filter_sizes']
        self.channels = architecture['channels']

        ih,iw,ic = architecture['initial_shape']
        self.initial_shape = [-1,ih,iw,ic]

        self.reshapes = []
        for reshape in architecture['reshapes']:
            rh,rw,rc = reshape
            self.reshapes.append([-1,rh,rw,rc])

        self.output_shape = self.reshapes[-1]
        self.final_output_shape = self.output_shape[1]*self.output_shape[2]

        self.up_convolution_counter = 1


    def add_layer(self, layer_input, filter_sz, channels, input_shape, output_shape):
        uc = self.up_convolution_counter # Which convolution is this?
        ih, iw, in_channels = input_shape
        _, oh, ow, oc = output_shape
        with tf.name_scope("Up_Convolution_%d" % (uc)):
            convWeightshape = [filter_sz[0], filter_sz[1], in_channels, channels]
            convWeights = tf.Variable(self.xavier_init(convWeightshape), dtype=self.dtype,
                    name='up_conv_weights%d' % (uc))
            print('input_shape = ', input_shape)
            bias = tf.Variable(tf.zeros([ih, iw, channels], dtype=self.dtype), name='bias%d' % (uc))

            print("bias.shape = ", bias.shape)
            print("layer_input.shape = ", layer_input.shape)
            print("convWeights.shape = ", convWeights.shape)

            conv = tf.nn.conv2d(layer_input, convWeights,
                                strides=[1,1,1,1],
                                padding='SAME',
                                name='up_conv%d' % (uc))

            print("conv.shape = ", conv.shape)

            conv_output = tf.nn.relu(tf.add(conv, bias), name='up_conv%d_output' % (uc))
            conv_output = tf.reshape(conv_output, output_shape)

        self.up_convolution_counter += 1
        return conv_output


    def _build_graph(self, network_input, input_shape, dtype=tf.float32, scope='UPCNN'):
        """The documentation for the build_graph routine of the UPCNN class. To come..."""

        _, ph, pw, pc = self.initial_shape # previous height, previous width, previous channels
        self.dtype = dtype
        # input in unwrapped format. Must reshape for convolution

        init_reshape_weights = tf.Variable(self.xavier_init((input_shape,np.prod([ph, pw, pc]))),
                dtype=self.dtype, name='init_reshape_weights')
        init_reshape_bias = tf.Variable(tf.zeros(np.prod([ph, pw, pc]), dtype=self.dtype),
                name='init_reshape_bias')

        print("network_input.shape = ", network_input.shape)
        print("init_reshape_weights.shape = ", init_reshape_weights.shape)
        tmp_current_input = tf.nn.relu(tf.add(tf.matmul(network_input, init_reshape_weights), init_reshape_bias),
                name='init_reshape')

        current_input = tf.reshape(tmp_current_input, (-1, ph, pw, pc))

        #init_reshape_weights = tf.Variable(self.xavier_init([input_shape, ph, pw, pc]),
                #dtype=self.dtype, name='init_reshape_weights')
        #init_reshape_bias = tf.Variable(tf.zeros([ph, pw, pc], dtype=self.dtype),
                #name='init_reshape_bias')

        #print("network_input.shape = ", network_input.shape)
        #print("init_reshape_weights.shape = ", init_reshape_weights.shape)
        #current_input = tf.nn.relu(tf.add(tf.tensordot(network_input, init_reshape_weights,
            #[[1], [0]]), init_reshape_bias), name='init_reshape')



        print("Up-Convolution")
        print("input_shape = ", input_shape)
        print("network_input.shape = ", network_input.shape)
        print("current_input.shape = ", current_input.shape)
        prev_r = [ph, pw, pc]
        with tf.name_scope(scope):
            for f,c,r in zip(self.filter_sizes, self.channels, self.reshapes):
                print("---------")
                print("filter = ", f)
                print("channels = ", c)
                print("reshape = ", r)
                current_input = self.add_layer(current_input, f, c, prev_r, r)
                print("current_input.shape = ", current_input.shape)
                prev_r = r[1:]

        final_output = tf.reshape(current_input, [-1,self.final_output_shape])

        return final_output


    def print_network_details(self):

        print("\ndtype = ", self.dtype)
        print("\ninput_shape = ", self.input_shape)
        print("\nchannels = ", self.channels)
        print("\nfilter_sizes = ", self.filter_sizes)
        print("\nstrides = ", self.strides)
        print("\noutput_sizes = ", self.output_shapes)
        print("\nfinal_output_shape = ", self.final_output_shape)
        print("\nfc_layer_size = ", self.fc_layer_size)


    def get_output_shape(self):

        return self.final_output_shape


class CNN3D(Neural_Network):


    def __init__(self, architecture):
        """ filter_sizes should be a list of tuples...
            [(f1_depth, f1_height, f1_width), (f2_depth, f2_height, f2_width), ... ]

            channels should be a list of integers...
            [8,16,13,...]

            len(filter_sizes) must equal len(channels)
        """

        Neural_Network.__init__(self, architecture)
        self.channels = architecture['channels']
        self.filter_sizes = architecture['filter_sizes']
        self.fc_layer_size = architecture['fc_layer_size']
        self.convolution_counter = 1
        assert len(self.filter_sizes) == len(self.channels)


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

            conv_output = tf.nn.relu(tf.add(conv, bias), name='3Dconv%d_output' % (cc))


        self.convolution_counter += 1
        return conv_output


    def _build_graph(self, network_input, dtype=tf.float32, scope='3DCNN'):
        """The documentation for the build_graph routine of the CNN class. To
        come..."""

        self.dtype = dtype
        img_d, img_h, img_w, img_c = input_shape
        current_input = tf.reshape(network_input, (-1, img_d, img_h, img_w, img_c))
        prevChannels = img_c
        with tf.name_scope(scope):
            for filterSz, numChannels in zip(self.filter_sizes,self.channels):
                current_input = self.add_layer(current_input, filterSz, prevChannels, numChannels)
                prevChannels = numChannels

        fcWeights = tf.Variable(self.xavier_init([img_d, img_h, img_w,
            prevChannels, self.fc_layer_size]), dtype=self.dtype, name='fc_weights')
        fcBias = tf.Variable(tf.zeros([self.fc_layer_size], dtype=self.dtype),
                name='fc_bias')

        fcOutput = tf.nn.relu(tf.add(tf.tensordot(current_input, fcWeights,
            [[1,2,3,4], [0,1,2,3]]), fcBias), name='fc_output')


        return fcOutput


class DCNN3D(Neural_Network):


    def __init__(self):
        None


    def _build_graph(self):
        None

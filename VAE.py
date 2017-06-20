from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf
import types
from IPython import embed
from tqdm import tqdm

# Requires Python 3.6+ and Tensorflow 1.1+


class Neural_Network():
    """Base class for various neural networks (consider using
       abstract base classes with 'from abc import ABCMeta')"""


    def __init__(self, architecture, transfer_funcs):

        # The below is hardly error proof. It is just a few minor checks to help
        # debugging. Eventually it will need to be bolstered further (check
        # architecture)
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


    def xavier_init(self, fan_in, fan_out, constant=1):
        """ Xavier initialization of network weights"""
        # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
        low = -constant*np.sqrt(6.0/(fan_in + fan_out))
        high = constant*np.sqrt(6.0/(fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out),
                                 minval=low, maxval=high,
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


class VAE():


    def __init__(self, input_dim, encoder, latency_dim, decoder, hyperParams):

        self.input_dim = input_dim
        self.encoder = encoder
        self.latency_dim = latency_dim
        self.decoder = decoder
        self.batch_size = hyperParams['batch_size'] # add error checking
        self.learning_rate = hyperParams['learning_rate'] # Add error checking
        if hyperParams['reconstruct_cost'] in ['bernoulli', 'gaussian']:
            self.reconstruct_cost = hyperParams['reconstruct_cost'] # Add error checking
        else:
            SystemExit("ERR: Only Gaussian and Bernoulli Reconstruction Functionality\n")
        self.optimizer = hyperParams['optimizer'] # Add error checking

        self.__build_graph()
        self.__create_loss()

        # Launch the session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    def __call__(self, network_input):

        _, cost = self.sess.run((self.train_op, self.cost), feed_dict={self.network_input: network_input})

        return cost


    def __build_graph(self):

        self.network_input = tf.placeholder(tf.float32, name="network_input")

        # Construct the encoder network and get its output
        encoder_output = self.encoder.build_graph(self.network_input, self.input_dim)
        #enc_output_dim = encoder_output.shape.as_list()[1]
        enc_output_dim = encoder.get_output_dim()

        # Now add the weights/bias for the mean and var of the latency dim
        z_mean_weight_val = self.encoder.xavier_init(enc_output_dim, self.latency_dim)
        z_mean_weight = tf.Variable(initial_value=z_mean_weight_val, dtype=tf.float32)
        z_mean_bias_val = np.zeros((1,self.latency_dim))
        z_mean_bias = tf.Variable(initial_value=z_mean_bias_val, dtype=tf.float32)

        self.z_mean = tf.nn.elu(encoder_output @ z_mean_weight + z_mean_bias)

        z_log_var_weight_val = self.encoder.xavier_init(enc_output_dim, self.latency_dim)
        z_log_var_weight = tf.Variable(initial_value=z_log_var_weight_val, dtype=tf.float32)
        z_log_var_bias_val = np.zeros((1,self.latency_dim))
        z_log_var_bias = tf.Variable(initial_value=z_log_var_bias_val, dtype=tf.float32)

        self.z_log_var = tf.nn.elu(encoder_output @ z_log_var_weight + z_log_var_bias)

        z_shape = tf.shape(self.z_log_var)
        eps = tf.random_normal(z_shape, dtype=tf.float32)
        self.z = self.z_mean + tf.sqrt(tf.exp(self.z_log_var)) * eps

        # Construct the decoder network and get its output
        decoder_output = self.decoder.build_graph(self.z, self.latency_dim)
        #dec_output_dim = decoder_output.shape.as_list()[1]
        dec_output_dim = decoder.get_output_dim()

        # Now add the weights/bias for the mean reconstruction terms
        x_mean_weight_val = self.encoder.xavier_init(dec_output_dim, self.input_dim)
        x_mean_weight = tf.Variable(initial_value=x_mean_weight_val, dtype=tf.float32)
        x_mean_bias_val = np.zeros(self.input_dim)
        x_mean_bias = tf.Variable(initial_value=x_mean_bias_val, dtype=tf.float32)

        # Just do Bernoulli for now. Add more functionality later
        if self.reconstruct_cost == 'bernoulli':
            self.x_mean = tf.nn.sigmoid(decoder_output @ x_mean_weight + x_mean_bias)
        elif self.reconstruct_cost == 'gaussian':
            self.x_mean = decoder_output @ x_mean_weight + x_mean_bias
            # Now add the weights/bias for the sigma reconstruction term
            x_sigma_weight_val = self.encoder.xavier_init(dec_output_dim, self.input_dim)
            x_sigma_weight = tf.Variable(initial_value=x_sigma_weight_val, dtype=tf.float32)
            x_sigma_bias_val = np.zeros(self.input_dim)
            x_sigma_bias = tf.Variable(initial_value=x_mean_bias_val, dtype=tf.float32)
            self.x_sigma = decoder_output @ x_sigma_weight + x_sigma_bias


    def __create_loss(self):

        if self.reconstruct_cost == "bernoulli":
            self.reconstruct_loss = \
                -tf.reduce_sum(self.network_input * tf.log(1e-10 + self.x_mean)
                               + (1-self.network_input) * tf.log(1e-10 + 1 -
                                   self.x_mean),1)
        elif self.reconstruct_cost == "gaussian":
            self.reconstruct_loss = tf.reduce_sum(tf.pow(tf.subtract(self.network_input,
                self.x_mean), 2))

        self.regularizer = -0.5 * tf.reduce_sum(1 + self.z_log_var
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_var), 1)

        self.cost = tf.reduce_mean(self.reconstruct_loss + self.regularizer)   # average over batch
        # User specifies optimizer in the hyperParams argument to constructor
        self.train_op = self.optimizer(learning_rate=self.learning_rate).minimize(self.cost)


    def reconstruct(self, network_input):

        if self.reconstruct_cost == 'bernoulli':
            return self.sess.run(self.x_mean, feed_dict={self.network_input: network_input})
        elif self.reconstruct_cost == 'gaussian':
            input_dict = {self.network_input: network_input}
            mean, sig = self.sess.run((self.x_mean, self.x_sigma), feed_dict=input_dict)
            eps = tf.random_normal(tf.shape(sigma), dtype=tf.float32)
            return mean + sigma * eps


    def getResponsibilities(self, x):
        None


if __name__ == "__main__":

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    input_dim = 784
    encoder = DNN([500]*2, tf.nn.elu)
    latency_dim = 10
    decoder = DNN([500]*2, tf.nn.elu)
    hyperParams = {'reconstruct_cost': 'bernoulli',
                   'learning_rate': 1e-4,
                   'optimizer': tf.train.AdamOptimizer,
                   'batch_size': 100}
    vae = VAE(input_dim, encoder, latency_dim, decoder, hyperParams)

    itrs_per_epoch = mnist.train.num_examples // hyperParams['batch_size']
    epochs = 5
    updates = 100
    cost = 0
    for itr in tqdm(range(epochs*itrs_per_epoch)):
        train_data, train_labels = mnist.train.next_batch(hyperParams['batch_size'])
        cost += vae(train_data)
        if itr % updates == 0:
            print("Avg Cost: %f" % (cost/(updates*hyperParams['batch_size'])))
            cost = 0


    import matplotlib.pyplot as plt
    test_data, test_labels = mnist.test.next_batch(hyperParams['batch_size'])
    reconstructions = vae.reconstruct(test_data)
    fig = plt.figure()
    for img in range(10):
        axes = fig.add_subplot(10,2,2*img+1)
        axes.imshow(test_data[img].reshape(28,28), cmap='gray')
        axes = fig.add_subplot(10,2,2*img+2)
        axes.imshow(reconstructions[img].reshape(28,28), cmap='gray')

    #plt.tight_layout()
    plt.show()

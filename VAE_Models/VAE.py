from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf
from IPython import embed

# Requires Python 3.6+ and Tensorflow 1.1+

class VAE():


    def __init__(self, input_dim, encoder, latent_dim, decoder, hyperParams):

        self.input_dim = input_dim
        self.encoder = encoder
        self.latent_dim = latent_dim
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

        targets = (self.cost, self.reconstruct_loss, self.regularizer, self.train_op)
        input_dict = {self.network_input: network_input}
        cost, reconstruct_loss, regularizer, _ = self.sess.run(targets, feed_dict=input_dict)

        return (cost, reconstruct_loss, regularizer)


    def __build_graph(self):

        self.network_input = tf.placeholder(tf.float32, name="network_input")

        # Construct the encoder network and get its output
        encoder_output = self.encoder.build_graph(self.network_input, self.input_dim)
        #enc_output_dim = encoder_output.shape.as_list()[1]
        enc_output_dim = self.encoder.get_output_dim()

        # Now add the weights/bias for the mean and var of the latency dim
        z_mean_weight_val = self.encoder.xavier_init(enc_output_dim, self.latent_dim)
        z_mean_weight = tf.Variable(initial_value=z_mean_weight_val, dtype=tf.float32)
        z_mean_bias_val = np.zeros((1,self.latent_dim))
        z_mean_bias = tf.Variable(initial_value=z_mean_bias_val, dtype=tf.float32)

        self.z_mean = encoder_output @ z_mean_weight + z_mean_bias

        z_log_var_weight_val = self.encoder.xavier_init(enc_output_dim, self.latent_dim)
        z_log_var_weight = tf.Variable(initial_value=z_log_var_weight_val, dtype=tf.float32)
        z_log_var_bias_val = np.zeros((1,self.latent_dim))
        z_log_var_bias = tf.Variable(initial_value=z_log_var_bias_val, dtype=tf.float32)

        self.z_log_var = encoder_output @ z_log_var_weight + z_log_var_bias

        z_shape = tf.shape(self.z_log_var)
        eps = tf.random_normal(z_shape, 0, 1, dtype=tf.float32)
        self.z = self.z_mean + tf.sqrt(tf.exp(self.z_log_var)) * eps

        # Construct the decoder network and get its output
        decoder_output = self.decoder.build_graph(self.z, self.latent_dim)
        #dec_output_dim = decoder_output.shape.as_list()[1]
        dec_output_dim = self.decoder.get_output_dim()

        # Now add the weights/bias for the mean reconstruction terms
        x_mean_weight_val = self.decoder.xavier_init(dec_output_dim, self.input_dim)
        x_mean_weight = tf.Variable(initial_value=x_mean_weight_val, dtype=tf.float32)
        x_mean_bias_val = np.zeros((1,self.input_dim))
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
            self.reconstruct_loss = tf.reduce_mean(tf.reduce_sum(
                    self.network_input * tf.log(1e-10 + self.x_mean) +
                    (1-self.network_input) * tf.log(1e-10 + 1 - self.x_mean),1))
        elif self.reconstruct_cost == "gaussian":
            reconstruct_loss = tf.reduce_sum(tf.square(self.network_input-self.x_mean))

        self.regularizer = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + self.z_log_var -
                    tf.square(self.z_mean) - tf.exp(self.z_log_var), axis=1))
        self.cost = -(self.reconstruct_loss - self.regularizer)

        # User specifies optimizer in the hyperParams argument to constructor
        self.train_op = self.optimizer(learning_rate=self.learning_rate).minimize(self.cost)


    def reconstruct(self, network_input):

        if self.reconstruct_cost == 'bernoulli':
            return self.sess.run(self.x_mean, feed_dict={self.network_input: network_input})
        elif self.reconstruct_cost == 'gaussian':
            input_dict = {self.network_input: network_input}
            targets = (self.x_mean, self.x_sigma)
            mean, sig = self.sess.run(targets, feed_dict=input_dict)
            eps = tf.random_normal(tf.shape(sigma), dtype=tf.float32)
            return mean + sigma * eps


    def transform(self, network_input):

        input_dict={self.network_input: network_input}
        targets = (self.z_mean, self.z_log_var)
        means, log_vars = self.sess.run(targets, feed_dict=input_dict)
        return (means, np.sqrt(np.exp(log_vars)))

    def generate(self, z=None):

        if z is None:
            z = np.random_normal((self.batch_size, self.latent_dim))
            return self.sess.run(self.x_mean, feed_dict={self.z: z})
        else:
            return self.sess.run(self.x_mean, feed_dict={self.z: z})


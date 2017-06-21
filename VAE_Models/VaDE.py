from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf
from .VAE import VAE
from IPython import embed

# Requires Python 3.6+ and Tensorflow 1.1+

class VaDE(VAE):


    def __init__(self, input_dim, encoder, latent_dim, decoder, hyperParams):

        self.num_clusters = hyperParams['num_clusters']
        VAE.__init__(self, input_dim, encoder, latent_dim, decoder, hyperParams)


    def __call__(self, network_input):

        targets = (self.cost, self.reconstruct_loss, self.regularizer, self.train_op)
        input_dict = {self.network_input: network_input}
        cost, reconstruct_loss, regularizer, _ = self.sess.run(targets, feed_dict=input_dict)

        return (cost, reconstruct_loss, regularizer)


    def __build_graph(self):


        # These values are not output from a network. They are variables
        # in the cost function. As a consequence they are learned during
        # the optimization procedure. So essentially, the network architecture
        # or framework is not different than a traditional VAE. Here we just
        # add extra variables and then learn them in the modified cost function
        pi_init = np.ones(self.num_clusters)/self.num_clusters
        self.gmm_pi = tf.Variable(pi_init, dtype=tf.float32)

        mu_init = np.zeros((self.latent_dim, self.num_clusters))
        self.gmm_mu = tf.Variable(mu_init, dtype=tf.float32)

        log_var_init = np.ones((self.latent_dim, self.num_clumster))
        self.gmm_log_var = tf.Variable(log_var_init, dtype=tf.float32)


        VAE.__build_graph()


    def __create_loss(self):

        # Reshape the GMM tensors in a frustratingly convoluted way to
        # be able to vectorize the computation of p(z|x) = E[p(c|z)]
        gmm_pi = tf.reshape(self.gmm_pi, (1,1,self.num_clusters))
        gmm_mu = tf.reshape(tf.tile(self.gmm_mu, [self.batch_size,1]),
                (self.batch_size,self.latent_dim,self.num_clusters))
        gmm_sigma = tf.reshape(tf.tile(self.gmm_sigma, [self.batch_size,1]),
                (self.batch_size,self.latent_dim,self.num_clusters))
        z = tf.reshape(self.z, (self.batch_size, self.latent_dim, 1))
        emebd()

        # First calculate the numerator p(c,z) = p(z|c)p(c) (vectorized)
        p_c_z = tf.exp(tf.reduce_sum(tf.log(gmm_pi) - \
            0.5*tf.log(2*np.pi*gmm_sigma) - tf.pow(z-gmm_mu,2)/(2*gmm_sigma),
            axis=1)) + 1e-10


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


    def transform(self, network_input):

        targets = (self.z_mean, self.z_log_var)
        return self.sess.run(target, feed_dict={self.network_input: network_input})


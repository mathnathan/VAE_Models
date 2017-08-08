from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf
from IPython import embed
import matplotlib.pyplot as plt
from tensorflow.contrib.tensorboard.plugins import projector
import os, sys

# Requires Python 3.6+ and Tensorflow 1.1+

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

  with tf.name_scope('Summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

class VAE():


    def __init__(self, input_dim, encoder, latent_dim, decoder, hyperParams,
            log_dir=None):

        self.CHECKPOINT_COUNTER = 0
        self.CALL_COUNTER = 0
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
        # Allow for other users on this GPU while training
        config.gpu_options.allow_growth=True
        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(tf.global_variables_initializer())

        # Use this to create internal saving and loading functionality
        self.saver = tf.train.Saver()
        # Store logs here. Checkpoints, models, tensorboard data
        if log_dir is None:
            log_dir = os.path.join(os.getcwd(), 'logs')
        self.LOG_DIR = log_dir
	# First start with logging the graph
        tf.summary.FileWriter(self.LOG_DIR, graph=self.sess.graph)
	# Now the summary statistics
        self.merged_summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.LOG_DIR+'/summaries')


    def __call__(self, network_input):
        """ Over load the parenthesis operator to act as a oneshot call for
            passing data through the network and updating with the optimizer

            network_input - (array) Input to the network. Typically of shape
            (batch_size, [input_dimensions])
        """

        input_dict = {self.network_input: network_input}
        # Log variables every 10 iterations
        if self.CALL_COUNTER % 10 == 0:
            targets = (self.merged_summaries, self.cost, self.reconstruct_loss,
                        self.regularizer, self.train_op)
            summary, cost, reconstruct_loss, regularizer, _ = \
                self.sess.run(targets, feed_dict=input_dict)
            self.summary_writer.add_summary(summary, self.CALL_COUNTER)
        else:
            targets = (self.cost, self.reconstruct_loss, self.regularizer, self.train_op)
            cost, reconstruct_loss, regularizer, _ = \
                self.sess.run(targets, feed_dict=input_dict)

        self.CALL_COUNTER += 1
        return (cost, reconstruct_loss, regularizer)


    def __build_graph(self):

        self.network_input = tf.placeholder(tf.float32, name='Input')

        with tf.name_scope('VAE'):

            # Construct the encoder network and get its output
            encoder_output = self.encoder.build_graph(self.network_input,
                    self.input_dim, scope='Encoder')
            #enc_output_dim = encoder_output.shape.as_list()[1]
            enc_output_dim = self.encoder.get_output_dim()

            # Now add the weights/bias for the mean and var of the latency dim
            z_mean_weight_val = self.encoder.xavier_init(enc_output_dim, self.latent_dim)
            z_mean_weight = tf.Variable(initial_value=z_mean_weight_val,
                    dtype=tf.float32, name='Z_Mean_Weight')
            z_mean_bias_val = np.zeros((1,self.latent_dim))
            z_mean_bias = tf.Variable(initial_value=z_mean_bias_val,
                    dtype=tf.float32, name='Z_Mean_Bias')

            self.z_mean = encoder_output @ z_mean_weight + z_mean_bias

            z_log_var_weight_val = self.encoder.xavier_init(enc_output_dim, self.latent_dim)
            z_log_var_weight = tf.Variable(initial_value=z_log_var_weight_val,
                    dtype=tf.float32, name='Z_Log_Var_Weight')
            z_log_var_bias_val = np.zeros((1,self.latent_dim))
            z_log_var_bias = tf.Variable(initial_value=z_log_var_bias_val,
                    dtype=tf.float32, name='Z_Log_Var_Bias')

            self.z_log_var = encoder_output @ z_log_var_weight + z_log_var_bias

            z_shape = tf.shape(self.z_log_var)
            eps = tf.random_normal(z_shape, 0, 1, dtype=tf.float32)
            self.z = self.z_mean + tf.sqrt(tf.exp(self.z_log_var)) * eps

            # Construct the decoder network and get its output
            decoder_output = self.decoder.build_graph(self.z, self.latent_dim,
                    scope='Decoder')
            #dec_output_dim = decoder_output.shape.as_list()[1]
            dec_output_dim = self.decoder.get_output_dim()

            # Now add the weights/bias for the mean reconstruction terms
            x_mean_weight_val = self.decoder.xavier_init(dec_output_dim, self.input_dim)
            x_mean_weight = tf.Variable(initial_value=x_mean_weight_val,
                    dtype=tf.float32, name='X_Mean_Weight')
            x_mean_bias_val = np.zeros((1,self.input_dim))
            x_mean_bias = tf.Variable(initial_value=x_mean_bias_val,
                    dtype=tf.float32, name='X_Mean_Bias')

            # Just do Bernoulli for now. Add more functionality later
            if self.reconstruct_cost == 'bernoulli':
                self.x_mean = tf.nn.sigmoid(decoder_output @ x_mean_weight + x_mean_bias)
            elif self.reconstruct_cost == 'gaussian':
                self.x_mean = tf.nn.sigmoid(decoder_output @ x_mean_weight + x_mean_bias)
                # Now add the weights/bias for the sigma reconstruction term
                x_sigma_weight_val = self.encoder.xavier_init(dec_output_dim, self.input_dim)
                x_sigma_weight = tf.Variable(initial_value=x_sigma_weight_val, dtype=tf.float32)
                x_sigma_bias_val = np.zeros(self.input_dim)
                x_sigma_bias = tf.Variable(initial_value=x_mean_bias_val, dtype=tf.float32)
                self.x_sigma = tf.nn.sigmoid(decoder_output @ x_sigma_weight +
                        x_sigma_bias)


    def __create_loss(self):

        with tf.name_scope('Calculate_Loss'):
            if self.reconstruct_cost == "bernoulli":
                with tf.name_scope('Bernoulli_Reconstruction'):
                    self.reconstruct_loss = tf.reduce_mean(tf.reduce_sum(
                            self.network_input * tf.log(1e-10 + self.x_mean) +
                            (1-self.network_input) * tf.log(1e-10 + 1 -
                                self.x_mean),1))
                    tf.summary.scalar('Reconstruction_Error', self.reconstruct_loss)
            elif self.reconstruct_cost == "gaussian":
                self.reconstruct_loss = tf.reduce_sum(tf.square(self.network_input-self.x_mean))

            with tf.name_scope('KL_Error'):
                self.regularizer = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + self.z_log_var -
                            tf.square(self.z_mean) - tf.exp(self.z_log_var), axis=1))
                tf.summary.scalar('KL_Loss', self.regularizer)
            with tf.name_scope('Total_Cost'):
                self.cost = -(self.reconstruct_loss - self.regularizer)
                tf.summary.scalar('Cost', self.cost)

            # User specifies optimizer in the hyperParams argument to constructor
            with tf.name_scope('Optimizer'):
                self.train_op = self.optimizer(learning_rate=self.learning_rate).minimize(self.cost)


    def reconstruct(self, network_input):

        with tf.name_scope('Reconstruct'):
            if self.reconstruct_cost == 'bernoulli':
                recon = self.sess.run(self.x_mean, feed_dict={self.network_input: network_input})
            elif self.reconstruct_cost == 'gaussian':
                input_dict = {self.network_input: network_input}
                targets = (self.x_mean, self.x_sigma)
                mean, sig = self.sess.run(targets, feed_dict=input_dict)
                eps = tf.random_normal(tf.shape(sig), dtype=tf.float32)
                recon = mean
            return recon

    def transform(self, network_input):

        with tf.name_scope('Transform'):
            input_dict={self.network_input: network_input}
            targets = (self.z_mean, self.z_log_var)
            means, log_vars = self.sess.run(targets, feed_dict=input_dict)
            return (means, np.sqrt(np.exp(log_vars)))


    def generate(self, z=None):

        with tf.name_scope('Generate'):
            if z is None:
                z = np.random_normal((self.batch_size, self.latent_dim))
            return self.sess.run(self.x_mean, feed_dict={self.z: z})


    def save(self, filename):

        with tf.name_scope('Save'):
            self.saver.save(self.sess, filename)


    def load(self, filename):

        with tf.name_scope('Load'):
            if os.path.isfile(filename):
                self.saver.restore(self.sess, filename)


    def create_embedding(self, batch, labels=None):
        """ This will eventually be called inside some convenient training
            routine that will be exposed to the user. This creates the logs
            and variables necessary to visualize the latency space with the
            embedding visualizer in tensorboard.

            batch - (array) Input to the network. Typically of shape
                    (batch_size, [input_dimensions])

            labels - (array) One dimensional array containing labels.
                     The element in the ith index of 'labels' is the
                     associated label for ith training element in batch
        """


        EMBED_VAR_NAME = "Latent_Space"
        EMBED_LOG_DIR = self.LOG_DIR
        SPRITES_PATH =  os.path.join(self.LOG_DIR, 'sprites.png')
        METADATA_PATH =  os.path.join(self.LOG_DIR, 'metadata.tsv')

        with tf.name_scope('Create_Embedding'):


        # ----- Create the embedding variable to be visualized -----
            # Paths to sprites and metadata must be relative to the location of
            # the projector config file which is determined by the line below
            embedding_writer = tf.summary.FileWriter(EMBED_LOG_DIR)

            latent_var = self.sess.run(self.z, feed_dict={self.network_input: batch})
            embedding_var = tf.Variable(latent_var, trainable=False, name=EMBED_VAR_NAME)
            # Initialize the newly created embedding variable
            init_embedding_var_op = tf.variables_initializer([embedding_var])
            self.sess.run(init_embedding_var_op)

            # Create a projector configuartion object
            config = projector.ProjectorConfig()
            # Create an embedding
            embedding = config.embeddings.add()
            # This is where the name is important again. Specify
            # which variable to embed
            embedding.tensor_name = embedding_var.name

            # Specify where you find the metadata
            embedding.metadata_path = METADATA_PATH #'metadata.tsv'

            # Specify where you find the sprite (we will create this later)
            embedding.sprite.image_path = SPRITES_PATH #'mnistdigits.png'
            embedding.sprite.single_image_dim.extend([28,28])

            # Say that you want to visualise the embeddings
            projector.visualize_embeddings(embedding_writer, config)

        # ----- Construct the Sprites for visualization -----
            images = np.reshape(batch, (-1,28,28))
            # Invert greyscale... Looks prettier in tensorboard
            images = 1-images
            batch_size, img_h, img_w = images.shape
            n_plots = int(np.ceil(np.sqrt(batch_size)))

            spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))

            for i in range(n_plots):
                for j in range(n_plots):
                    this_filter = i * n_plots + j
                    if this_filter < images.shape[0]:
                        this_img = images[this_filter]
                        spriteimage[i * img_h:(i + 1) * img_h,
                          j * img_w:(j + 1) * img_w] = this_img

            plt.imsave(SPRITES_PATH, spriteimage, cmap='gray')


        # ----- Create the metadata file for visualization -----
            if not labels is None:
                with open(METADATA_PATH,'w') as f:
                    f.write("Index\tLabel\n")
                    for index,label in enumerate(labels):
                        f.write("%d\t%d\n" % (index,label))

        embed_saver = tf.train.Saver()
        embed_saver.save(self.sess, os.path.join(EMBED_LOG_DIR, "embed.ckpt"), self.CHECKPOINT_COUNTER)

        self.CHECKPOINT_COUNTER += 1

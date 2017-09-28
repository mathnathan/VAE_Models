from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import tensorflow as tf
from IPython import embed
import matplotlib.pyplot as plt
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python import debug as tf_debug
import os, sys

tf.set_random_seed(12345)
np.random.seed(12345)

DEBUG = 0

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


    def __init__(self, input_shape, encoder, latent_dim, decoder, hyperParams,
            initializers={}, logdir=None):

        self.CHECKPOINT_COUNTER = 0
        self.CALL_COUNTER = 0
        self.input_shape = input_shape
        self.num_input_vals = np.prod(input_shape)
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.decoder = decoder
        self.__parse_hyperParams(hyperParams)
        self.initializers = dict(initializers)
        self.__build_graph()
        self.__create_loss()

        # Launch the session
        config = tf.ConfigProto()
        # Allow for other users on this GPU while training
        config.gpu_options.allow_growth=True
        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(tf.global_variables_initializer())
        if DEBUG:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # Use this to create internal saving and loading functionality
        self.saver = tf.train.Saver()
        # Store logs here. Checkpoints, models, tensorboard data
        if logdir is None:
            logdir = os.path.join(os.getcwd(), 'logs')
        self.LOG_DIR = logdir
	# First start with logging the graph
        tf.summary.FileWriter(self.LOG_DIR, graph=self.sess.graph)
	# Now the summary statistics
        self.merged_summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.LOG_DIR+'/summaries')


    def __parse_hyperParams(self, hyperParams):

        try:
            self.batch_size = hyperParams['batch_size']
            self.optimizer = hyperParams['optimizer']
            self.starter_learning_rate = hyperParams['learning_rate']
        except KeyError as key:
            raise KeyError('%s must be specified in hyperParams' % (key)) from key

        try:
            self.prior = hyperParams['prior']
            if not (self.prior in ['gaussian', 'gmm']):
                raise ValueError("No functionality for %s prior. Only \
                        'gaussian' and 'gmm' priors are supported" % (key)) from key
        except KeyError as key:
            print("\n--- Prior not specified... Defaulting to 'gaussian'. Specify with 'prior' parameter")
            self.prior = 'gaussian'
        else:
            try:
                self.num_clusters = hyperParams['num_clusters']
            except KeyError as key:
                if self.prior == 'gmm':
                    print("\n--- Number of GMM Modes not specified... Defaulting to 10. Specify with 'num_clusters' parameter")
                    self.num_clusters = 10
        try:
            self.reconstruct_cost = hyperParams['reconstruct_cost']
            if not (self.reconstruct_cost in ['bernoulli', 'gaussian']):
                raise ValueError("Only 'gaussian' and 'bernoulli' reconstruction functionality supported\n")
        except KeyError:
            print("\n--- Reconstruction Cost not specified... Defaulting to 'gaussian'. Specify with 'reconstruct_cost' parameter")
            self.reconstruct_cost = 'gaussian'

        try:
            self.variational = hyperParams['variational']
        except:
            self.variational = True

        try:
            self.alpha = hyperParams['alpha']
        except:
            self.alpha = 1.0

        try:
            self.decay_step = hyperParams['decay_step']
        except:
            # If not specified set it so high that there is no decay
            self.decay_step = 1e14

        try:
            self.decay_rate = hyperParams['decay_rate']
        except:
            print("\n--- Decay Rate not specified... Defaulting to 0.9. Specify with 'decay_rate' parameter")
            self.decay_rate = 0.9

        # Used for exponential learning rate decay
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate,
                                                        self.global_step,
                                                        self.decay_step,
                                                        self.decay_rate)


    def __call__(self, network_input):
        """ Over load the parenthesis operator to act as a oneshot call for
            passing data through the network and updating with the optimizer

            network_input - (array) Input to the network. Typically of shape
            (batch_size, input_dimensions)

            returns - (total cost, reconstruction loss, KL loss)
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


        print("\nBuilding VAE")
        self.network_input = tf.placeholder(tf.float32, name='Input')

        with tf.name_scope('VAE'):

            if self.prior == 'gmm':
                # These values are not output from a network. They are variables
                # in the cost function. As a consequence they are learned during
                # the optimization procedure. So essentially, the network architecture
                # or framework is not different than a traditional VAE. Here we just
                # add extra variables and then learn them in the modified cost function
                # This is only for the GMM prior
                if 'gmm_pi' in self.initializers:
                    pi_init = self.initializers['gmm_pi']
                else:
                    pi_init = np.ones(self.num_clusters)/self.num_clusters
                self.gmm_pi = tf.Variable(pi_init, dtype=tf.float32)
                tf.summary.histogram('gmm_pi', self.gmm_pi)

                if 'gmm_mu' in self.initializers:
                    mu_init = self.initializers['gmm_mu']
                else:
                    means = np.zeros(self.latent_dim)
                    cov = np.eye(self.latent_dim)
                    mu_init = np.random.multivariate_normal(means, cov, self.num_clusters)
                #self.gmm_mu = tf.Variable(mu_init.T, dtype=tf.float32)
                self.gmm_mu = tf.Variable(mu_init, dtype=tf.float32)
                tf.summary.histogram('gmm_mu', self.gmm_mu)

                if 'gmm_log_var' in self.initializers:
                    log_var_init = self.initializers['gmm_log_var']
                else:
                    log_var_init = np.ones((self.num_clusters, self.latent_dim))
                self.gmm_log_var = tf.Variable(log_var_init, dtype=tf.float32)
                tf.summary.histogram('gmm_log_var', self.gmm_log_var)


            # Construct the encoder network and get its output
            encoder_output = self.encoder.build_graph(self.network_input,
                    self.input_shape, scope='Encoder')
            #enc_output_dim = encoder_output.shape.as_list()[1]
            enc_output_dim = self.encoder.get_output_dim()

            # Now add the weights/bias for the mean and var of the latency dim
            z_mean_weight_val = self.encoder.xavier_init((enc_output_dim,
                self.latent_dim))
            z_mean_weight = tf.Variable(initial_value=z_mean_weight_val,
                    dtype=tf.float32, name='Z_Mean_Weight')
            z_mean_bias_val = np.zeros((1,self.latent_dim))
            z_mean_bias = tf.Variable(initial_value=z_mean_bias_val,
                    dtype=tf.float32, name='Z_Mean_Bias')

            self.z_mean = encoder_output @ z_mean_weight + z_mean_bias

            z_log_var_weight_val = self.encoder.xavier_init((enc_output_dim,
                self.latent_dim))
            z_log_var_weight = tf.Variable(initial_value=z_log_var_weight_val,
                    dtype=tf.float32, name='Z_Log_Var_Weight')
            z_log_var_bias_val = np.zeros((1,self.latent_dim))
            z_log_var_bias = tf.Variable(initial_value=z_log_var_bias_val,
                    dtype=tf.float32, name='Z_Log_Var_Bias')

            self.z_log_var = encoder_output @ z_log_var_weight + z_log_var_bias

            z_shape = tf.shape(self.z_log_var)
            eps = tf.random_normal(z_shape, 0, 1, dtype=tf.float32)
            self.z = self.z_mean + tf.sqrt(tf.exp(self.z_log_var)) * eps if self.variational else self.z_mean

            # Construct the decoder network and get its output
            decoder_output = self.decoder.build_graph(self.z, self.latent_dim,
                    scope='Decoder')
            #dec_output_dim = decoder_output.shape.as_list()[1]
            dec_output_dim = self.decoder.get_output_dim()

            # Now add the weights/bias for the mean reconstruction terms
            x_mean_weight_val = self.decoder.xavier_init((dec_output_dim,
                self.num_input_vals))
            x_mean_weight = tf.Variable(initial_value=x_mean_weight_val,
                    dtype=tf.float32, name='X_Mean_Weight')
            x_mean_bias_val = np.zeros((1,self.num_input_vals))
            x_mean_bias = tf.Variable(initial_value=x_mean_bias_val,
                    dtype=tf.float32, name='X_Mean_Bias')

            if self.reconstruct_cost == 'bernoulli':
                self.x_mean = tf.nn.sigmoid(decoder_output @ x_mean_weight + x_mean_bias)
            elif self.reconstruct_cost == 'gaussian':
                self.x_mean = tf.nn.sigmoid(decoder_output @ x_mean_weight + x_mean_bias)
                # Now add the weights/bias for the sigma reconstruction term
                x_sigma_weight_val = self.encoder.xavier_init((dec_output_dim,
                    self.num_input_vals))
                x_sigma_weight = tf.Variable(initial_value=x_sigma_weight_val, dtype=tf.float32)
                x_sigma_bias_val = np.zeros(self.num_input_vals)
                x_sigma_bias = tf.Variable(initial_value=x_mean_bias_val, dtype=tf.float32)
                self.x_sigma = tf.nn.sigmoid(decoder_output @ x_sigma_weight +
                        x_sigma_bias)


    def __create_loss(self):

        with tf.name_scope('Calculate_Loss'):
            if self.prior == 'gaussian':
                if self.reconstruct_cost == "bernoulli":
                    with tf.name_scope('Bernoulli_Reconstruction'):
                        self.reconstruct_loss = tf.reduce_mean(tf.reduce_sum(
                                self.network_input * tf.log(1e-10 + self.x_mean) +
                                (1-self.network_input) * tf.log(1e-10 + 1 -
                                    self.x_mean),1))
                elif self.reconstruct_cost == "gaussian":
                    with tf.name_scope('Gaussian_Reconstruction'):
                        if self.variational:
                            self.reconstruct_loss = tf.reduce_mean(tf.square(self.network_input-self.x_mean))
                        else:
                            self.reconstruct_loss = -tf.reduce_mean(tf.square(self.network_input-self.x_mean))
                tf.summary.scalar('Reconstruction_Error', self.reconstruct_loss)

                with tf.name_scope('KL_Error'):
                    self.regularizer = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + self.z_log_var -
                                tf.square(self.z_mean) - tf.exp(self.z_log_var), axis=1))
                    tf.summary.scalar('KL_Loss', self.regularizer)
                with tf.name_scope('Total_Cost'):
                    self.cost = -(self.reconstruct_loss - self.alpha*self.regularizer)
                    tf.summary.scalar('Cost', self.cost)

                # User specifies optimizer in the hyperParams argument to constructor
                with tf.name_scope('Optimizer'):
                    #sess = tf.InteractiveSession() # --- TO BE REMOVED
                    #sess.run(tf.global_variables_initializer()) # --- TO BE REMOVED
                    #network_input = np.random.rand(100,784)
                    #print("deconv output = ", sess.run(self.decoder_output,
                    #    feed_dict={self.network_input: network_input}))
                    self.train_op = self.optimizer(self.learning_rate).minimize(self.cost,
                            global_step=self.global_step)

            elif self.prior == 'gmm':
                # Reshape everything to be compatible with broadcasting for
                # dimensions of (batch_size, num_clusters, latent_dim)
                # Except gmm_pi, it is used infrequently and the below reshaping is enough
                #reshaped_gmm_pi = tf.reshape(self.gmm_pi, (1,self.num_clusters))
                #exp_gmm_pi = tf.exp(reshaped_gmm_pi)
                #gmm_pi = tf.divide(exp_gmm_pi, tf.reduce_sum(exp_gmm_pi, axis=1), name='gmm_pi')
                gmm_pi = tf.reshape(self.gmm_pi, (1,self.num_clusters), name='gmm_pi')

                gmm_mu = tf.reshape(self.gmm_mu, (1,self.num_clusters,self.latent_dim),
                        name='gmm_mu')
                gmm_log_var = tf.reshape(self.gmm_log_var,(1,self.num_clusters,self.latent_dim),
                        name='gmm_log_var')
                z = tf.reshape(self.z, (self.batch_size, 1, self.latent_dim),
                        name='z')
                z_mean = tf.reshape(self.z_mean, (self.batch_size, 1,
                    self.latent_dim), name='z_mean')
                z_log_var = tf.reshape(self.z_log_var, (self.batch_size, 1,
                    self.latent_dim), name='z_log_var')

                with tf.name_scope('Determine_p_c_z'):
                    # First calculate the numerator p(c,z) = p(c)p(z|c) (vectorized)
                    # resulting shape = (batch_size, num_clusters)
                    tf.summary.histogram('z', z)
                    p_cz = tf.exp(tf.log(1e-10+gmm_pi)
                            - 0.5*(tf.reduce_sum(tf.log(2*np.pi)
                            + gmm_log_var + tf.square(z-gmm_mu)
                            / tf.exp(gmm_log_var), axis=2)), name='p_cz')
                    tf.summary.histogram('p_cz', p_cz)

                    # Next we sum over the clusters making the marginal probability p(z)
                    p_z = tf.reduce_sum(p_cz, axis=1, keep_dims=True)
                    tf.summary.scalar('p_z', tf.reduce_mean(p_z))

                    # Finally we calculate the resulting posterior p(c|z), in GMM clustering
                    # literature this is called the 'responsibility' and is denoted by a
                    # gamma - shape = (batch_size, num_clusters)
                    self.gamma = tf.divide(p_cz, 1e-10+p_z, name='gamma')
                    tf.summary.histogram('gamma', self.gamma)


                if self.reconstruct_cost == 'bernoulli':
                    with tf.name_scope('Bernoulli_Reconstruction'):
                        # E[log p(x|z)]
                        p_x_z = tf.reduce_mean(tf.reduce_sum(self.network_input *
                                tf.log(1e-10 + self.x_mean)
                                + (1-self.network_input)
                                * tf.log(1e-10 + 1 - self.x_mean),
                                axis=1, name='p_x_z'))
                elif self.reconstruct_cost == 'gaussian':
                    with tf.name_scope('Gaussian_Reconstruction'):
                        # E[log p(x|z)]
                        p_x_z = tf.reduce_mean(tf.square(self.network_input-self.x_mean),
                                name='p_x_z')

                tf.summary.scalar('E_p_x_z', p_x_z)

                with tf.name_scope('Total_Cost'):

                    # E[log p(z|c)]
                    p_z_c = tf.reduce_mean(-tf.reduce_sum(self.gamma
                            * (0.5*self.latent_dim*tf.log(2*np.pi)
                            + 0.5*tf.reduce_sum(gmm_log_var
                            + tf.exp(z_log_var)/tf.exp(gmm_log_var)
                            + tf.square(z_mean-gmm_mu)/tf.exp(gmm_log_var),
                            axis=2)), axis=1))
                    tf.summary.scalar('E_p_z_c', p_z_c)

                    # E[log p(c)]
                    p_c = tf.reduce_mean(tf.reduce_sum(self.gamma*tf.log(1e-10+gmm_pi), axis=1))
                    tf.summary.scalar('E_p_c', p_c)

                    # E[log q(z|x)]
                    q_z_x = tf.reduce_mean(-0.5*(self.latent_dim*tf.log(2*np.pi)
                            + tf.reduce_sum(1 + z_log_var, axis=2)))
                    tf.summary.scalar('E_q_z_x', q_z_x)

                    # E[log q(c|x)]
                    q_c_x = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(self.gamma
                            * tf.log(1e-10+self.gamma),1)))
                    tf.summary.scalar('E_q_c_x', q_c_x)

                    self.cost = -(p_x_z + p_z_c + p_c - q_z_x - q_c_x)
                tf.summary.scalar('Cost', self.cost)

                self.reconstruct_loss = -p_x_z
                tf.summary.scalar('Reconstruction_Error', self.reconstruct_loss)
                self.regularizer = self.reconstruct_loss - self.cost
                tf.summary.scalar('KL_Loss', self.regularizer)

                # User specifies optimizer in the hyperParams argument to constructor
                self.train_op = self.optimizer(self.learning_rate).minimize(self.cost,
                        global_step=self.global_step)

                # Ensure modes are normalized
                #self.normalize_pis_op = tf.assign(self.gmm_pi,self.gmm_pi/tf.reduce_sum(self.gmm_pi))


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
            if self.variational:
                targets = (self.z_mean, self.z_log_var)
                means, log_vars = self.sess.run(targets, feed_dict=input_dict)
                return (means, np.sqrt(np.exp(log_vars)))
            else:
                targets = self.z_mean
                means = self.sess.run(targets, feed_dict=input_dict)
                return means


    def generate(self, z=None):

        with tf.name_scope('Generate'):
            if self.prior == 'gaussian':
                if z is None:
                    z = np.random_normal((self.batch_size, self.latent_dim))
                return self.sess.run(self.x_mean, feed_dict={self.z: z})

            elif self.prior == 'gmm':
                if z is None:
                    targets = (self.gmm_pi, self.gmm_mu, self.gmm_log_var)
                    pis, means, log_vars = self.sess.run(targets)
                    pis /= pis.sum()
                    cluster = np.random.choice(range(self.num_clusters), p=pis)
                    mean = means[cluster]
                    std = np.sqrt(np.exp(log_vars[cluster]))
                    eps = np.random_normal(std.shape)
                    z = mean + std * eps
                return self.sess.run(self.x_mean, feed_dict={self.z: z})


    def save(self, filename):

        with tf.name_scope('Save'):
            self.saver.save(self.sess, filename)


    def load(self, filename):

        with tf.name_scope('Load'):
            if os.path.isfile(filename):
                self.saver.restore(self.sess, filename)


    def get_gmm_params(self):

        if self.prior == 'gmm':
            targets = (self.gmm_pi, self.gmm_mu, self.gmm_log_var)
            pis, means, log_vars = self.sess.run(targets)
            pis /= pis.sum()
            return (pis, means, np.sqrt(np.exp(log_vars)))
        elif self.prior == 'gaussian':
            return None


    def predict_clusters(self, input_x):

        input_dict = {self.network_input: input_x}
        targets = (self.q_c_x) # Probability of each cluster given x. aka responsibility
        predictions = self.sess.run(targets, feed_dict=input_dict)
        return predictions


    def create_embedding(self, batch, img_shape, labels=None, invert_colors=True):
        """ This will eventually be called inside some convenient training
            routine that will be exposed to the user. This creates the logs
            and variables necessary to visualize the latency space with the
            embedding visualizer in tensorboard.

            batch - (array) Input to the network. Typically of shape
                    (batch_size, [input_dimensions])

            img_shape - (array like) Can be an array, tuple or list containing
                        the dimensions of one image in (height, width) format.
                        For example, if using MNIST img_shape might be equal
                        to (28,28) or [28,28]

            labels - (array) One dimensional array containing labels.
                     The element in the ith index of 'labels' is the
                     associated label for ith training element in batch
        """


        EMBED_VAR_NAME = "Latent_Space"
        EMBED_LOG_DIR = self.LOG_DIR
        SPRITES_PATH =  os.path.join(self.LOG_DIR, 'sprites.png')
        METADATA_PATH =  os.path.join(self.LOG_DIR, 'metadata.tsv')
        img_h, img_w = img_shape

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
            embedding.sprite.single_image_dim.extend([img_h,img_w])

            # Say that you want to visualise the embeddings
            projector.visualize_embeddings(embedding_writer, config)

        # ----- Construct the Sprites for visualization -----
            images = np.reshape(batch, (-1,img_h,img_w))
            # Maybe invert greyscale... MNIST looks prettier in tensorboard
            if invert_colors:
                images = 1-images
            batch_size = images.shape[0]
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
            with open(METADATA_PATH,'w') as f:
                if labels is None:
                    for index in range(batch_size):
                        f.write("%d\n" % (index))
                else:
                    f.write("Index\tLabel\n")
                    for index,label in enumerate(labels):
                        f.write("%d\t%d\n" % (index,label))


        embed_saver = tf.train.Saver()
        embed_saver.save(self.sess, os.path.join(EMBED_LOG_DIR, "embed.ckpt"), self.CHECKPOINT_COUNTER)

        self.CHECKPOINT_COUNTER += 1

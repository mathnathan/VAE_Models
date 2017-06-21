import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from IPython import embed
import sys

from VAE_Models.VAE import VAE
from VAE_Models.architectures import DNN
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


sys.path.append("../libraries")
from vae import VariationalAutoencoder
def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=1):

    n_samples = 55000
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)
    # Training cycle
    for epoch in tqdm(range(training_epochs)):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in tqdm(range(total_batch)):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs, batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    return vae

network_architecture = dict(n_hidden_recog=[500]*2, # num neurons in each layer of encoder network
                        n_hidden_gener=[500]*2,  # num neurons in each layer of generator network
                        n_input=784, # MNIST data input (img shape: 28*28)
                        n_z=2)  # dimensionality of latent space

vae = train(network_architecture, training_epochs=35, batch_size=100)
test_data, test_labels = mnist.test.next_batch(100)
reconstructions = vae.reconstruct(test_data)
fig = plt.figure()
for img in range(10):
    axes = fig.add_subplot(10,2,2*img+1)
    axes.imshow(test_data[img].reshape(28,28), cmap='gray')
    axes = fig.add_subplot(10,2,2*img+2)
    axes.imshow(reconstructions[img].reshape(28,28), cmap='gray')

plt.show()
sys.exit()

# Here we define the structure of the VAE. We can pick and choose different
# architectures to plug in as the encoder and decoder. The glueing together
# of the encoder and decoder is done inside of the VAE class. This will need
# to be the case for any new vae architectures that we code up, i.e. VaDE.
input_dim = 784
encoder = DNN([500]*2, tf.nn.elu)
latency_dim = 2
decoder = DNN([500]*2, tf.nn.elu)
hyperParams = {'reconstruct_cost': 'bernoulli',
               'learning_rate': 1e-4,
               'optimizer': tf.train.AdamOptimizer,
               'batch_size': 100}

vae = VAE(input_dim, encoder, latency_dim, decoder, hyperParams)

itrs_per_epoch = mnist.train.num_examples // hyperParams['batch_size']
epochs = 35
updates = 1000
cost = 0
reconstruct_cost = 0
kl_cost = 0
#for itr in tqdm(range(epochs*itrs_per_epoch)):
for itr in range(epochs*itrs_per_epoch):
    train_data, train_labels = mnist.train.next_batch(hyperParams['batch_size'])
    tot_cost, reconstr_loss, KL_loss = vae(train_data)
    cost += tot_cost
    reconstruct_cost += reconstr_loss
    kl_cost += KL_loss
    if itr % updates == 0:
        c = cost/(updates*hyperParams['batch_size'])
        rl = reconstruct_cost/(updates*hyperParams['batch_size'])
        kl = kl_cost/(updates*hyperParams['batch_size'])
        print("recon_loss=%f, KL_loss=%f, cost=%f\n" % (rl,kl,c))
        cost = 0
        reconstruct_cost = 0
        kl_cost = 0


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

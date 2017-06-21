import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from IPython import embed

from VAE_Models.VAE import VAE
from VAE_Models.architectures import DNN
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Here we define the structure of the VAE. We can pick and choose different
# architectures to plug in as the encoder and decoder. The glueing together
# of the encoder and decoder is done inside of the VAE class. This will need
# to be the case for any new vae architectures that we code up, i.e. VaDE.
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
epochs = 2
updates = 1000
cost = 0
for itr in tqdm(range(epochs*itrs_per_epoch)):
    train_data, train_labels = mnist.train.next_batch(hyperParams['batch_size'])
    cost += vae(train_data)
    #if itr % updates == 0:
        #print("Avg Cost: %f" % (cost/(updates*hyperParams['batch_size'])))
        #cost = 0


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

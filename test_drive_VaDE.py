import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from IPython import embed

from VAE_Models.VaDE import VaDE
from VAE_Models.architectures import DNN
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Here we define the structure of the VAE. We can pick and choose different
# architectures to plug in as the encoder and decoder. The glueing together
# of the encoder and decoder is done inside of the VAE class. This will need
# to be the case for any new vae architectures that we code up, i.e. VaDE.
input_dim = 784
encoder = DNN([10,2000,500,500], tf.nn.elu)
latency_dim = 2
decoder = DNN([500,500,2000,10], tf.nn.elu)
hyperParams = {'reconstruct_cost': 'bernoulli',
               'learning_rate': 1e-4,
               'optimizer': tf.train.AdamOptimizer,
               'batch_size': 100,
               'alpha': 1.0,
               'num_clusters': 10}

vade = VaDE(input_dim, encoder, latency_dim, decoder, hyperParams)

itrs_per_epoch = mnist.train.num_examples // hyperParams['batch_size']
epochs = 150
updates = 1000
cost = 0
reconstruct_cost = 0
kl_cost = 0
#for itr in tqdm(range(epochs*itrs_per_epoch)):
for itr in range(epochs*itrs_per_epoch):
    train_data, train_labels = mnist.train.next_batch(hyperParams['batch_size'])
    tot_cost, reconstr_loss, KL_loss = vade(train_data)
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
reconstructions = vade.reconstruct(test_data)
fig = plt.figure()
for img in range(10):
    axes = fig.add_subplot(10,2,2*img+1)
    axes.imshow(test_data[img].reshape(28,28), cmap='gray')
    axes = fig.add_subplot(10,2,2*img+2)
    axes.imshow(reconstructions[img].reshape(28,28), cmap='gray')

#plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from IPython import embed
import sys

# Choose standard VAE or VaDE
#from VAE_Models.VaDE import VaDE as model
from VAE_Models.VAE import VAE as model
from VAE_Models.architectures import DNN
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#print_tensors_in_checkpoint_file(file_name='logs/embed.ckpt-0', tensor_name='Create_Embedding/Latent_Space', all_tensors=False)
#sys.exit()

# ---- First Estimate the GMM parameters using standard
# ---- autoencoder and GMM clustering

#from sklearn import mixture

#gmm = mixture.GaussianMixture(10, covatiance_type='diag')


# Here we define the structure of the VAE. We can pick and choose different
# architectures to plug in as the encoder and decoder. The glueing together
# of the encoder and decoder is done inside of the VAE class. This will need
# to be the case for any new vae architectures that we code up, i.e. VaDE.
input_dim = (28,28,1)
encoder = DNN([512])
latent_dim = 10
decoder = DNN([512])
hyperParams = {'reconstruct_cost': 'gaussian',
               'variational': False,
               'prior': 'gaussian',
               'learning_rate': 1e-3,
               'optimizer': tf.train.AdamOptimizer,
               'batch_size': 100}

network = model(input_dim, encoder, latent_dim, decoder, hyperParams)

itrs_per_epoch = mnist.train.num_examples // hyperParams['batch_size']
epochs = 10

test_data, test_labels = mnist.train.next_batch(hyperParams['batch_size'])
for itr in tqdm(range(epochs*itrs_per_epoch)):
    train_data, train_labels = mnist.train.next_batch(hyperParams['batch_size'])
    tot_cost, reconstr_loss, KL_loss = network(train_data)

test_data, test_labels = mnist.test.next_batch(1000)
network.create_embedding(test_data, (28,28), np.where(test_labels)[1])

reconstructions = network.reconstruct(test_data)
fig = plt.figure()
numFigs = 5
for img in range(numFigs):
    axes = fig.add_subplot(numFigs,2,2*img+1)
    axes.imshow(test_data[img].reshape(28,28))
    axes = fig.add_subplot(numFigs,2,2*img+2)
    axes.imshow(reconstructions[img].reshape(28,28))

#plt.tight_layout()
plt.show()

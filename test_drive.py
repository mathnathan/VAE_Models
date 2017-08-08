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
input_dim = 784
encoder = DNN([1024,512,256], tf.nn.elu)
latency_dim = 64
decoder = DNN([256,512,1024], tf.nn.elu)
hyperParams = {'reconstruct_cost': 'bernoulli',
               'learning_rate': 1e-4,
               'optimizer': tf.train.AdamOptimizer,
               'batch_size': 100}
#               'alpha': 1.0,
#               'num_clusters': 10}

network = model(input_dim, encoder, latency_dim, decoder, hyperParams)

itrs_per_epoch = mnist.train.num_examples // hyperParams['batch_size']
epochs = 100
updates = 1000
cost = 0
reconstruct_cost = 0
kl_cost = 0
winSize = 28
filename = 'figures'

#for itr in tqdm(range(epochs*itrs_per_epoch)):
nx = ny = 20
winRange = 10
canvas = np.empty((winSize*ny,winSize*nx))
numPlots = 0
x_values = np.linspace(-winRange, winRange, nx)
y_values = np.linspace(-winRange, winRange, ny)
interval = 1000 # save fig of latency space every 10 batches

img_cost = 0.0
test_data, test_labels = mnist.test.next_batch(1000)
test_labels = [np.where(arr == 1) for arr in test_labels]
for itr in tqdm(range(epochs*itrs_per_epoch)):
    train_data, train_labels = mnist.train.next_batch(hyperParams['batch_size'])
    tot_cost, reconstr_loss, KL_loss = network(train_data)
    cost += tot_cost
    img_cost += tot_cost
    reconstruct_cost += reconstr_loss
    kl_cost += KL_loss
    if itr % updates == 0:
        c = cost/(updates*hyperParams['batch_size'])
        rl = reconstruct_cost/(updates*hyperParams['batch_size'])
        kl = kl_cost/(updates*hyperParams['batch_size'])
        print("\nrecon_loss=%f, KL_loss=%f, cost=%f" % (rl,kl,c))
        cost = 0
        reconstruct_cost = 0
        kl_cost = 0
        if hasattr(network, 'get_gmm_params'):
            pi, mean, std = network.get_gmm_params()
            print("Modes\n", pi)
            print("Means\n", mean)

test_data, test_labels = mnist.test.next_batch(1000)
network.create_embedding(test_data, np.where(test_labels)[1])

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

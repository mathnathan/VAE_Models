import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import mixture
from sklearn.manifold import TSNE
import pickle
from VAE_Models.VAE import VAE as model
from VAE_Models.architectures import DNN
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
datasets = [mnist.train, mnist.test, mnist.validation]
dataset_size = sum([ds.num_examples for ds in datasets])
dataset_probs = [ds.num_examples/dataset_size for ds in datasets]

input_dim = (28,28)
encoder = DNN([500,500,2000], tf.nn.relu)
latency_dim = 10
decoder = DNN([2000,500,500], tf.nn.relu)
hyperParams = {'reconstruct_cost': 'gaussian',
               'learning_rate': 0.002,
               'optimizer': tf.train.AdamOptimizer,
               'batch_size': 100,
               'num_clusters': 10,
               'prior': 'gaussian',
               'alpha': 0,
               'variational': False}


itrs_per_epoch = dataset_size // hyperParams['batch_size']
hyperParams['decay_steps'] = 10*itrs_per_epoch
hyperParams['decay_rate'] = 0.9
epochs = 20

# Create a standard autoencoder by specifying prior=gaussian, alpha=0,
# variational=False, and reconstruct_cost=gaussian
AE = model(input_dim, encoder, latency_dim, decoder, hyperParams, logdir='ae_logs')

# Pretrain a simple stacked autoencoder
for itr in tqdm(range(epochs*itrs_per_epoch)):
    ds = np.random.choice(datasets, p=dataset_probs)
    train_data, train_labels = ds.next_batch(hyperParams['batch_size'])
    tot_cost, reconstr_loss, KL_loss = AE(train_data)

all_data = []
for ds in datasets:
    all_data.extend(ds.next_batch(ds.num_examples)[0])
#latent_vecs = AE.transform(all_data)
latent_vecs = AE.transform(all_data)

gmm = mixture.GaussianMixture(hyperParams['num_clusters'], covariance_type='diag')
gmm.fit(latent_vecs)
initializers = {}
initializers['gmm_pi'] = gmm.weights_
initializers['gmm_mu'] = gmm.means_.T
initializers['gmm_log_var'] = np.log(gmm.covariances_).T

pickle_file = open('initializers.pkl', 'wb')
pickle.dump(initializers, pickle_file)

validation_data, validation_labels = mnist.validation.next_batch(1000)
labels = np.where(validation_labels)[1]
AE.create_embedding(validation_data, (28,28), labels)

if 0:
    reconstructions = AE.reconstruct(validation_data)
    fig = plt.figure(0)
    numFigs = 8
    for img in range(numFigs):
        axes = fig.add_subplot(numFigs,2,2*img+1)
        axes.imshow(validation_data[img].reshape(28,28))
        axes = fig.add_subplot(numFigs,2,2*img+2)
        axes.imshow(reconstructions[img].reshape(28,28))

    lvs = AE.transform(validation_data)
    gmm_preds = gmm.predict(lvs)
    counter = 1
    for perp in tqdm([15,20,25,30]):
        mappings = TSNE(n_components=2, perplexity=perp).fit_transform(lvs)
        plt.figure(counter)
        plt.title("AE Representation with Perplexity = %d" % (perp))
        plt.scatter(mappings[:,0], mappings[:,1], c=labels)
        plt.figure(counter+1)
        plt.title("GMM Representation with Perplexity = %d" % (perp))
        plt.scatter(mappings[:,0], mappings[:,1], c=gmm_preds)
        counter += 2

    plt.show()

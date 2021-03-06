import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from IPython import embed
import sys, os
from sklearn.manifold import TSNE
import pickle

def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size,ind


init_types = ['random', 'approx_nc', 'approx_vade', 'perfect']
init = init_types[2]
# Choose standard VAE or VaDE
#from VAE_Models.VaDE import VaDE as model
from VAE_Models.VAE import VAE as model
from VAE_Models.architectures import DNN
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
datasets = [mnist.train, mnist.test, mnist.validation]
dataset_size = sum([ds.num_examples for ds in datasets])
dataset_probs = [ds.num_examples/dataset_size for ds in datasets]

tf.set_random_seed(12345)
np.random.seed(12345)

FILENAME = 'exps/mnist_vade1/exp'
input_dim = (28,28)
encoder = DNN([500,500,2000], tf.nn.relu)
latency_dim = 10
decoder = DNN([2000,500,500], tf.nn.relu)
hyperParams = {'reconstruct_cost': 'bernoulli',
               'learning_rate': 0.002,
               'optimizer': tf.train.AdamOptimizer,
               'batch_size': 100,
               'num_clusters': 10,
               'prior': 'gmm',
               'variational': True}

itrs_per_epoch = dataset_size // hyperParams['batch_size']
hyperParams['decay_steps'] = 10*itrs_per_epoch
hyperParams['decay_rate'] = 0.9
epochs = 300

if init == 'random':
    VaDE = model(input_dim, encoder, latency_dim, decoder, hyperParams, logdir='vade_logs')
elif init == 'approx_nc':
    initializers = pickle.load(open('initializers.pkl', 'rb'))
    VaDE = model(input_dim, encoder, latency_dim, decoder,
            hyperParams, initializers, logdir='vade_logs')
elif init == 'approx_vade':
    initializers = {}
    #initializers['gmm_pi'] = np.load('pretrain_params/theta_p.npy')
    num_clusts = hyperParams['num_clusters']
    initializers['gmm_pi'] = np.ones(num_clusts)/num_clusts
    initializers['gmm_mu'] = np.load('pretrain_params/mu.npy').T
    initializers['gmm_log_var'] = np.log(np.load('pretrain_params/lambda.npy').T)
    VaDE = model(input_dim, encoder, latency_dim, decoder,
            hyperParams, initializers, logdir='vade_logs')
elif init == 'perfect':
    initializers = {}
    initializers['gmm_pi'] = np.load('pretrain_params/theta_p.npy')
    initializers['gmm_mu'] = np.load('pretrain_params/u_p.npy')
    initializers['gmm_log_var'] = np.log(np.load('pretrain_params/lambda_p.npy'))
    VaDE = model(input_dim, encoder, latency_dim, decoder,
            hyperParams, initializers, logdir='vade_logs')



#pi,mu,std = VaDE.get_gmm_params()
#print('init[gmm_pi] = ', np.round(initializers['gmm_pi'], 2))
#print("pis = ", np.round(pi,2))
#print('init[gmm_mu] = ', np.round(initializers['gmm_mu'], 2))
#print("mu = ", np.round(mu,2))
#print('init[gmm_log_var] = ', np.round(initializers['gmm_log_var'], 2))
#print('sqrt(exp(init[gmm_log_var])) = ', np.round(np.sqrt(np.exp(initializers['gmm_log_var'])),2))
#print("std = ", np.round(std,2))


if os.path.exists(FILENAME+'.meta'):
    VaDE.load(FILENAME)
else:
    for itr in tqdm(range(epochs*itrs_per_epoch)):
        ds = np.random.choice(datasets, p=dataset_probs)
        data, labels = ds.next_batch(hyperParams['batch_size'])
        if itr % itrs_per_epoch == 0:

            enumerate_labels = np.where(labels)[1]
            preds = VaDE.predict_clusters(data)
            acc = cluster_acc(np.argmax(preds,axis=1), enumerate_labels)
            print("Accuracy: ", acc[0])
        tot_cost, reconstr_loss, KL_loss = VaDE(data)
    VaDE.save(FILENAME)

validation_data, validation_labels = mnist.validation.next_batch(1000)
labels = np.where(validation_labels)[1]
VaDE.create_embedding(validation_data, (28,28), labels)

means, stds = VaDE.transform(validation_data)
latent_vecs = means + np.random.normal(size=stds.shape)*stds
reconstructions = VaDE.reconstruct(validation_data)
fig1 = plt.figure(0)
numFigs = 5
for img in range(numFigs):
    axes = fig1.add_subplot(numFigs,2,2*img+1)
    axes.imshow(validation_data[img].reshape(28,28))
    axes = fig1.add_subplot(numFigs,2,2*img+2)
    axes.imshow(reconstructions[img].reshape(28,28))


if 1:
    counter = 1
    for perp in tqdm([15,20,25,30]):
        mappings = TSNE(n_components=2, perplexity=perp).fit_transform(latent_vecs)
        plt.figure(counter)
        plt.title("Representation with Perplexity = %d" % (perp))
        plt.scatter(mappings[:,0], mappings[:,1], c=labels)
        counter += 1

plt.show()

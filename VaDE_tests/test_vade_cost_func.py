import tensorflow as tf
import numpy as np
import sys

input_shape = 5
num_z_samples = 3
batch_size = 2
latent_dim = 3
num_clusters = 4

network_input = tf.constant([[.2,.1,1.,.2,0],[.3,.1,0,.9,.2]])
x_mean = tf.constant([[.1,.1,1.,.2,.1],[.3,0,0,.9,.1]])


# ------ First look at calculating the responsibilities ------
# E[log p(x|z)]
p_x_z = tf.reduce_mean(tf.reduce_sum(network_input *
        tf.log(1e-7 + x_mean)
        + (1.0-network_input)
        * tf.log(1e-7 + (1.0 - x_mean)),
        axis=1), name='p_x_z')

if 0:
    sess = tf.Session()
    print("---Bernoulli Reconstruction Loss---\n")
    print("-network_input-\n", sess.run(network_input))
    print("\n-x_mean-\n", sess.run(x_mean))
    print("\n-tf.log(1e-10 + x_mean)-\n", sess.run(tf.log(1e-10 + x_mean)))
    print("\n-network_input * tf.log(1e-10 + x_mean)-\n", sess.run(network_input * tf.log(1e-10 + x_mean)))
    print("\n-(1.0-network_input)-\n",sess.run(1.0-network_input))
    print("\n-(tf.log(1e-10 + (1.0 - x_mean))-\n",sess.run(tf.log(1e-10 + (1.0 - x_mean))))
    print("\n-(tf.log(1e-10 + 1.0 - x_mean)-\n",sess.run(tf.log(1e-10 + 1.0 - x_mean)))
    print("\n-(1.0-network_input) * tf.log(1e-10 + 1.0 - x_mean)-\n",sess.run(1.0-network_input * tf.log(1e-10 + (1.0 - x_mean))))
    print("\n-tf.reduce_sum((1.0-network_input) * tf.log(1e-10 + (1.0 - x_mean)),axis=1)-\n",sess.run(tf.reduce_sum(1.0-network_input * tf.log(1e-10 + (1.0 - x_mean)),axis=1)))
    print("\np_x_z = ", sess.run(p_x_z))
    sys.exit()

z_mean = tf.constant([[2,1,2],[0,-3,1]], dtype=tf.float32)
z_log_var = tf.constant([[-2,-1,-1],[-3,-3,-2]], dtype=tf.float32)

# Take multiple samples from latency space to calculate the
# q(c|x) = E[p(c|z)]
new_z_shape = (num_z_samples, batch_size, latent_dim, 1)
eps = tf.ones(new_z_shape, dtype=tf.float32)
z_mean_rs = tf.reshape(z_mean, (1,batch_size,latent_dim,1))
z_log_var_rs = tf.reshape(z_log_var, (1,batch_size,latent_dim,1))
z = z_mean_rs + tf.sqrt(tf.exp(z_log_var_rs)) * eps

if 0:
    sess = tf.Session()
    print('---Reshaping Z variables---\n')
    print('-eps-\n', sess.run(eps))
    print('-z_mean_rs-\n', sess.run(z_mean_rs))
    print('-z_log_var_rs-\n', sess.run(z_log_var_rs))
    print('-exp(z_log_var_rs)-\n', sess.run(tf.exp(z_log_var_rs)))
    print('-sqrt(exp(z_log_var_rs))-\n', sess.run(tf.sqrt(tf.exp(z_log_var_rs))))
    print('-z-\n', sess.run(z))
    sys.exit()

# These reshapes are for broadcasting along the
# new z samples axis
gmm_mu = tf.constant([[-1,2,0,1],[3,2,-2,-2],[0,0,0,-2]], dtype=tf.float32)
gmm_log_var = tf.constant([[-3,-2,-1,-1],[-2,0,0,-1],[-3,-1,0,-1]], dtype=tf.float32)
gmm_pi = tf.constant([0.05,0.6,0.2,0.15], dtype=tf.float32)

pcz_gmm_mu = tf.reshape(gmm_mu, (1,1,latent_dim, num_clusters))
pcz_gmm_log_var = tf.reshape(gmm_log_var, (1,1,latent_dim, num_clusters))
pcz_gmm_pi = tf.reshape(gmm_pi, (1,1,num_clusters))

# First calculate the numerator p(c,z) = p(c)p(z|c) (vectorized)
# sum over the latent dim, axis=2
# resulting shape = (num_z_samples, batch_size, num_clusters)

#p_cz = tf.exp(latent_dim*tf.log(1e-10+pcz_gmm_pi)
#        - 0.5*(tf.reduce_sum(tf.log(2*np.pi)
#        + pcz_gmm_log_var + tf.square(z-pcz_gmm_mu)
#        / tf.exp(pcz_gmm_log_var), axis=2)), name='p_cz')
p_cz = tf.exp(tf.reduce_sum(tf.log(1e-10+pcz_gmm_pi)
        - 0.5*(tf.log(2*np.pi) + pcz_gmm_log_var + tf.square(z-pcz_gmm_mu) / tf.exp(pcz_gmm_log_var)), axis=2), name='p_cz')

# Next we sum over the clusters making the marginal probability p(z)
p_z = tf.reduce_sum(p_cz, axis=2, keep_dims=True, name='p_z_var')

# Finally we calculate the resulting posterior
# q(c|x)=E[p(c|z)]=E[p(c,z)/sum_c[p(c,z)]]. Take the mean over the
# new z samples axis for the expectation. In GMM clustering
# literature this is called the 'responsibility' and is
# denoted by a gamma - shape = (batch_size, num_clusters)
gamma = tf.reduce_mean(p_cz/(1e-10+p_z), axis=0, name='gamma')

if 0:
    sess = tf.Session()
    print('---Calculate p(c,z)---\n')
    print('\n-pcz_gmm_pi-\n', sess.run(pcz_gmm_pi))
    print('\n-pcz_gmm_log_var-\n', sess.run(pcz_gmm_log_var))
    print('\n-z-\n', sess.run(z))
    print('\n-pcz_gmm_mu-\n', sess.run(pcz_gmm_mu))
    print('\n-tf.log(1e-7+pcz_gmm_pi)-\n', sess.run(tf.log(1e-7+pcz_gmm_pi)))
    print('\n-z-pcz_gmm_mu-\n', sess.run(z-pcz_gmm_mu))
    print('\n-tf.square(z-pcz_gmm_mu)-\n', sess.run(tf.square(z-pcz_gmm_mu)))
    print('\n-tf.exp(pcz_gmm_log_var)-\n', sess.run(tf.exp(pcz_gmm_log_var)))
    print('\n-tf.square(z-pcz_gmm_mu)/tf.exp(pcz_gmm_log_var)-\n', sess.run(tf.square(z-pcz_gmm_mu)/tf.exp(pcz_gmm_log_var)))
    print('\n-pcz_gmm_log_var + tf.square(z-pcz_gmm_mu)/tf.exp(pcz_gmm_log_var)-\n', sess.run(pcz_gmm_log_var + tf.square(z-pcz_gmm_mu)/tf.exp(pcz_gmm_log_var)))
    print('\n-tf.reduce_sum(pcz_gmm_log_var + tf.square(z-pcz_gmm_mu)/tf.exp(pcz_gmm_log_var),axis=2)-\n', sess.run(tf.reduce_sum(pcz_gmm_log_var + tf.square(z-pcz_gmm_mu)/tf.exp(pcz_gmm_log_var),axis=2)))
    print('\n-p_cz-\n', sess.run(p_cz))
    print('\n-p_z-\n', sess.run(p_z))
    print('\n-gamma-\n', sess.run(gamma))

# ------ First look at calculating the responsibilities ------


# ------ Next look at calculating the cost function ------

# Reshape everything to be compatible with broadcasting for
# dimensions of (batch_size, latent_dim, num_clusters)
#reshaped_gmm_pi = tf.reshape(gmm_pi, (1,1,num_clusters))
#exp_gmm_pi = tf.exp(reshaped_gmm_pi)
#gmm_pi = tf.divide(exp_gmm_pi, tf.reduce_sum(exp_gmm_pi, axis=1), name='gmm_pi')

gmm_pi = tf.reshape(gmm_pi, (1,num_clusters))
gmm_mu = tf.reshape(gmm_mu, (1,latent_dim, num_clusters))
gmm_log_var = tf.reshape(gmm_log_var,(1,latent_dim,num_clusters))
z_mean = tf.reshape(z_mean, (batch_size, latent_dim, 1))
z_log_var = tf.reshape(z_log_var, (batch_size, latent_dim, 1))

# E[log p(z|c)]
p_z_c = tf.reduce_mean(-0.5*tf.reduce_sum(gamma
        * (latent_dim*tf.log(2*np.pi)
        + tf.reduce_sum(gmm_log_var
        + tf.exp(z_log_var)/tf.exp(gmm_log_var)
        + tf.square(z_mean-gmm_mu)/tf.exp(gmm_log_var),
        axis=1)), axis=1))

if 0:
    sess = tf.Session()
    print('\n-p_z_c-\n', sess.run(p_z_c))

# E[log p(c)]
p_c = tf.reduce_mean(tf.reduce_sum(gamma*tf.log(1e-10+gmm_pi), axis=1))
if 1:
    sess = tf.Session()
    print('\n---E[log p(c)]---\n')
    print('\n-gamma-\n', sess.run(gamma))
    print('\n-gmm_pi-\n', sess.run(gmm_pi))
    print('\n-tf.log(1e-10 + gmm_pi)-\n', sess.run(tf.log(1e-10 + gmm_pi)))
    print('\n-gamma * tf.log(1e-10 + gmm_pi)-\n', sess.run(gamma * tf.log(1e-10 + gmm_pi)))
    print('\n-tf.reduce_sum(gamma * tf.log(1e-10 + gmm_pi),axis=1)-\n', sess.run(tf.reduce_sum(gamma * tf.log(1e-10 + gmm_pi), axis=1)))
    print('\n-p_c-\n', sess.run(p_c))
    sys.exit()


# E[log q(z|x)]
q_z_x = tf.reduce_mean(-0.5*(latent_dim*tf.log(2*np.pi)
        + tf.reduce_sum(1.0 + z_log_var, axis=1)))


# E[log q(c|x)]
q_c_x = tf.reduce_mean(tf.reduce_sum(gamma * tf.log(1e-10+gamma),axis=1))

if 0:
    sess = tf.Session()
    print('\n---E[log q(c|x)]---\n')
    print('\n-gamma-\n', sess.run(gamma))
    print('\n-tf.log(1e-10 + gamma)-\n', sess.run(tf.log(1e-10 + gamma)))
    print('\n-gamma * tf.log(1e-10 + gamma)-\n', sess.run(gamma * tf.log(1e-10 + gamma)))
    print('\n-tf.reduce_sum(gamma * tf.log(1e-10 + gamma),axis=1)-\n', sess.run(tf.reduce_sum(gamma * tf.log(1e-10 + gamma), axis=1)))
    print('\n-q_c_x-\n', sess.run(q_c_x))
    sys.exit()

cost = -(p_x_z + p_z_c + p_c - q_z_x - q_c_x)
#cost = -p_x_z

reconstruct_loss = -p_x_z
regularizer = reconstruct_loss - cost


print('\n--Cost--\n', sess.run(cost))

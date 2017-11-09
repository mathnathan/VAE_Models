import numpy as np


def q_z_x(z, z_mean, z_log_var):
    return -0.5*(latent_dim*np.log(2*np.pi) + np.sum(1.0 + z_log_var))

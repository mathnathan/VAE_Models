import tensorflow as tf


def tf_huber(x,delta=1):
    quad_where = tf.where(x<=delta)
    linear_where = tf.where(x>delta)
    quad_vals = tf.gather_nd(x,quad_where)
    linear_vals = tf.gather_nd(x,linear_where)
    
    quad_vals = .5*tf.pow(quad_vals,2)
    linear_vals = delta*(tf.abs(linear_vals) - .5 * delta)
    
    x_new_quad = tf.scatter_nd(quad_where,quad_vals,x.shape)
    x_new_linear = tf.scatter_nd(linear_where,linear_vals,x.shape)
    return(x_new_quad+x_new_linear)

import tflib as lib

import numpy as np
import tensorflow as tf

def Layernorm(name, norm_axes, inputs):
    mean, var = tf.nn.moments(inputs, norm_axes, keep_dims=True)

    # Assume the 'neurons' axis is the first of norm_axes. This is the case for fully-connected and BCHW conv layers.
    n_neurons = inputs.get_shape().as_list()[norm_axes[0]]

    offset = lib.param(name+'.offset', np.zeros(n_neurons, dtype='float32'))
    scale = lib.param(name+'.scale', np.ones(n_neurons, dtype='float32'))

    # Add broadcasting dims to offset and scale (e.g. BCHW conv data)
    offset = tf.reshape(offset, [-1] + [1 for i in xrange(len(norm_axes)-1)])
    scale = tf.reshape(scale, [-1] + [1 for i in xrange(len(norm_axes)-1)])

    result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)

    return result
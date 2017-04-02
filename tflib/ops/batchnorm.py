import tflib as lib

import numpy as np
import tensorflow as tf

def Batchnorm(name, axes, inputs, is_training=None, stats_iter=None, update_moving_stats=True, fused=True):
    if ((axes == [0,2,3]) or (axes == [0,2])) and fused==True:
        if axes==[0,2]:
            inputs = tf.expand_dims(inputs, 3)
        # Old (working but pretty slow) implementation:
        ##########

        # inputs = tf.transpose(inputs, [0,2,3,1])

        # mean, var = tf.nn.moments(inputs, [0,1,2], keep_dims=False)
        # offset = lib.param(name+'.offset', np.zeros(mean.get_shape()[-1], dtype='float32'))
        # scale = lib.param(name+'.scale', np.ones(var.get_shape()[-1], dtype='float32'))
        # result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-4)

        # return tf.transpose(result, [0,3,1,2])

        # New (super fast but untested) implementation:
        offset = lib.param(name+'.offset', np.zeros(inputs.get_shape()[1], dtype='float32'))
        scale = lib.param(name+'.scale', np.ones(inputs.get_shape()[1], dtype='float32'))

        moving_mean = lib.param(name+'.moving_mean', np.zeros(inputs.get_shape()[1], dtype='float32'), trainable=False)
        moving_variance = lib.param(name+'.moving_variance', np.ones(inputs.get_shape()[1], dtype='float32'), trainable=False)

        def _fused_batch_norm_training():
            return tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=1e-5, data_format='NCHW')
        def _fused_batch_norm_inference():
            # Version which blends in the current item's statistics
            batch_size = tf.cast(tf.shape(inputs)[0], 'float32')
            mean, var = tf.nn.moments(inputs, [2,3], keep_dims=True)
            mean = ((1./batch_size)*mean) + (((batch_size-1.)/batch_size)*moving_mean)[None,:,None,None]
            var = ((1./batch_size)*var) + (((batch_size-1.)/batch_size)*moving_variance)[None,:,None,None]
            return tf.nn.batch_normalization(inputs, mean, var, offset[None,:,None,None], scale[None,:,None,None], 1e-5), mean, var

            # Standard version
            # return tf.nn.fused_batch_norm(
            #     inputs,
            #     scale,
            #     offset,
            #     epsilon=1e-2, 
            #     mean=moving_mean,
            #     variance=moving_variance,
            #     is_training=False,
            #     data_format='NCHW'
            # )

        if is_training is None:
            outputs, batch_mean, batch_var = _fused_batch_norm_training()
        else:
            outputs, batch_mean, batch_var = tf.cond(is_training,
                                                       _fused_batch_norm_training,
                                                       _fused_batch_norm_inference)
            if update_moving_stats:
                no_updates = lambda: outputs
                def _force_updates():
                    """Internal function forces updates moving_vars if is_training."""
                    float_stats_iter = tf.cast(stats_iter, tf.float32)

                    update_moving_mean = tf.assign(moving_mean, ((float_stats_iter/(float_stats_iter+1))*moving_mean) + ((1/(float_stats_iter+1))*batch_mean))
                    update_moving_variance = tf.assign(moving_variance, ((float_stats_iter/(float_stats_iter+1))*moving_variance) + ((1/(float_stats_iter+1))*batch_var))

                    with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                        return tf.identity(outputs)
                outputs = tf.cond(is_training, _force_updates, no_updates)

        if axes == [0,2]:
            return outputs[:,:,:,0] # collapse last dim
        else:
            return outputs
    else:
        # raise Exception('old BN')
        # TODO we can probably use nn.fused_batch_norm here too for speedup
        mean, var = tf.nn.moments(inputs, axes, keep_dims=True)
        shape = mean.get_shape().as_list()
        if 0 not in axes:
            print "WARNING ({}): didn't find 0 in axes, but not using separate BN params for each item in batch".format(name)
            shape[0] = 1
        offset = lib.param(name+'.offset', np.zeros(shape, dtype='float32'))
        scale = lib.param(name+'.scale', np.ones(shape, dtype='float32'))
        result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)


        return result

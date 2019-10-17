import tensorflow as tf
import logging
import pdb
import numpy as np

logging.error(tf.__version__)

batch_size = 3
zdim = 8
shuffle_mask = [tf.constant(np.random.choice(np.arange(batch_size),batch_size,False)) for i in range(zdim)]
params = tf.reshape(tf.range(batch_size*zdim,dtype=tf.float32),[batch_size,zdim])
shuffled_params = []
for z in np.arange(zdim):
    shuffled_params.append(tf.gather(params[:,z],shuffle_mask[z],axis=0))
shuffled_params = tf.stack(shuffled_params,axis=-1)
pdb.set_trace()

sess = tf.Session()
params = sess.run(params)
shuffle_mask = sess.run(shuffle_mask)
shuffled_params = sess.run(shuffled_params)
pdb.set_trace()

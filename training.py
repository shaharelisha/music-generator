# Code based on Dan Shiebler's RBM music generator: https://github.com/dshieble/Music_RBM

import tensorflow as tf
import numpy as np
import rbm
import midi_manipulation
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm


epochs_to_save = 5 # Number of epochs to run between saving each checkpoint

num_timesteps = rbm.num_timesteps
x, emotions, W, bh, bv = rbm.get_variables()
lr = rbm.lr           # learning rate
num_epochs = rbm.num_epochs

def sample(probs):
    """
    Given a vector of probabilities, each element is added to a random value between 0 and 1, and then 
    rounded down to give a value of either 0 or 1. Higher input probabilities have a higher chance of being 1.

    :param probs: vector of probablities
    :type probs: tensor
    :returns: vector of 0s and 1s sampled from the input vector
    :rtype: tensor
    """
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))
    

def gibbs_sample(k):
    """
    Runs k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv.

    :params k: number of gibbs step iterations to run
    :type k: int
    :returns: matrix of music sampled
    :rtype: tensor
    """
    def gibbs_step(count, k, xk):
        """
        Runs a single gibbs step
        :param count: number of iterations done
        :param k: total number of gibbs step iterations to run
        :param xk: the visible values are initialized to xk.
        :type count: int
        :type k: int
        :type xk: tensor
        """
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh)) # Propagate the visible values to sample the hidden values
        tv = tf.matmul(hk, tf.transpose(W)) + bv       # Propagate the hidden values to the visible values
        xk = sample(tf.sigmoid(tv))                    # sample the visible values       
        return count+1, k, xk

    # Run gibbs steps for k iterations
    ct = tf.constant(0) # initialize counter to 0
    [_, _, x_sample] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter,
                                         gibbs_step, [ct, tf.constant(k), x], 1, False)
    # This is not strictly necessary in this implementation, but if you want to adapt this code to use one of TensorFlow's
    #optimizers, you need this in order to stop tensorflow from propagating gradients back through the gibbs step
    x_sample = tf.stop_gradient(x_sample) 
    return x_sample


x_sample = gibbs_sample(1) 
# The sample of the hidden nodes, starting from the known visible state of x (ground truth)
h = sample(tf.sigmoid(tf.matmul(x, W) + bh)) 
# The sample of the hidden nodes, starting from the visible state of x_sample
h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh)) 

# update the values of W, bh, and bv, based on the difference between the samples that we drew and the original values
size_bt = tf.cast(tf.shape(x)[0], tf.float32)
W_adder  = tf.mul(lr/size_bt, tf.sub(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
bv_adder = tf.mul(lr/size_bt, tf.reduce_sum(tf.sub(x, x_sample), 0, True))
bh_adder = tf.mul(lr/size_bt, tf.reduce_sum(tf.sub(h, h_sample), 0, True))
# When we call sess.run(updt), TensorFlow will run the following 3 update steps
updt = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]

songs = midi_manipulation.get_songs('Midi_Files')       # list of songs in matrix form
print "{} songs processed".format(len(songs))

saver = tf.train.Saver(max_to_keep=None) 

with tf.Session() as sess:
    # initialize the variables of the model
    init = tf.initialize_all_variables()
    sess.run(init)
    # Run through all of the training data num_epochs times
    for epoch in tqdm(range(num_epochs)):
        for song in songs:
            # The songs are stored in a time x notes format. The size of each song is total_timesteps x 2*note_range+2
            # reshape the songs so that each training example is a vector with num_timesteps x 2*note_range+2
            song = np.array(song)
            rows = song.shape[0]/(num_timesteps)
            song = song[:rows*num_timesteps]
            
            # split songs into smaller sections of 64 timesteps
            song = np.reshape(song, [rows, song.shape[1]*(num_timesteps)])
            sub_songs = np.vsplit(song, rows)

            for s in sub_songs:
                sess.run(updt, feed_dict={x: s})

        # Save the weights and biases of the model every few epochs
        if (epoch + 1) % epochs_to_save == 0:
            saver.save(sess, "parameter_checkpoints/epoch", global_step=epoch)

    save_path = saver.save(sess, "parameter_checkpoints/trained_system")        

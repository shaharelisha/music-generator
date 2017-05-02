# Code based on Dan Shiebler's RBM music generator: https://github.com/dshieble/Music_RBM

import tensorflow as tf
import numpy as np
import rbm
from tensorflow.python.ops import control_flow_ops
import midi_manipulation
from tqdm import tqdm


num_timesteps = rbm.num_timesteps
x, emotions, W, bh, bv = rbm.get_variables()

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


def sample_correct(probs, emotions):
	"""
	Given a vector of probabilities, each element is added to a random value between 0 and 1, and then 
	rounded down to give a value of either 0 or 1. Higher input probabilities have a higher chance of being 1.

	In addition to sampling, the emotion values are clamped to the last two elements of each timestep.

	:param probs: vector of probablities
	:param emotions: vector of 8 emotion values (4 arousal, 4 valence)
	:type probs: tensor
	:type emotions: tensor
	:returns: vector of 0s and 1s sampled from the input vector
	:rtype: tensor
	"""
	notes = tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))[0]
	notes = tf.unpack(notes)
	emotions = tf.unpack(emotions)
	for i in range(16):
		j=156
		e = i / 4 * 2
		notes[j] = emotions[e]
		notes[j+1] = emotions[e+1]
		j += 158
	notes = tf.pack(notes)
	final = tf.cast(notes, tf.float32)
	final = tf.reshape(final, [1, 10112])
	return final


def gibbs_sample_generate(k):
	"""
	Runs k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv.

	:params k: number of gibbs step iterations to run
	:type k: int
	:returns: matrix of music sampled
	:rtype: tensor
	"""
	def gibbs_step_generate(count, k, xk):
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
		tv = tf.matmul(hk, tf.transpose(W)) + bv	   # Propagate the hidden values to the visible values
		xk = sample_correct(tf.sigmoid(tv), emotions)  # sample the visible values and clamp emotion values
		return count+1, k, xk

	x_sample = x
	# run gibbs steps for k iterations
	ct = tf.constant(0) # initialize counter to 0
	[_, _, x_sample] = control_flow_ops.While(lambda count, num_iter, *args: count < num_iter,
										 gibbs_step_generate, [ct, tf.constant(k), x_sample], 1, False)
	# This is not strictly necessary in this implementation, but if you want to adapt this code to use one of TensorFlow's
	# optimizers, you need this in order to stop tensorflow from propagating gradients back through the gibbs step
	x_sample = tf.stop_gradient(x_sample) 
	return x_sample


def generate_music(emotion_text_file, midi_file_name):
	"""
	Given a series of emotion data points, the function will generate music that corresponds
	to those emotion values. The system variables (weights and biases) will be loaded from a 
	pre-trained model. Iterating through the emotion values, they're assigned to a matrix and
	sent through the RBM to generate music. The music is saved as a midi file. 

	:param emotion_text_file: path to text file holding emotion data
	:param midi_file_name: name for midi file
	:type emotion_text_file: str
	:type midi_file_name: str
	:returns: none 
	:rtype: none
	"""
	saved_weights_path = "parameter_checkpoints/trained_system"	  # path to saved, trained model
	trainable_vars = [W, bh, bv]
	n_visible = rbm.n_visible
	note_range = midi_manipulation.span

	saver = tf.train.Saver(trainable_vars) # restore the weights and biases of the model

	with tf.Session() as sess:
		init = tf.initialize_all_variables()
		sess.run(init)
		saver.restore(sess, saved_weights_path) # load the saved weights and biases of the model
		
		text = open(emotion_text_file, 'r')		# open text file with emotion values
		x_ = np.zeros((1, n_visible))
		j=156
		emotions_ = np.array([])

		# iterate through each line of the emotion text file
		for l, line in enumerate(text):
			emotion = line[:-1].split(",")
			a = float(emotion[0])		# arousal value
			v = float(emotion[1])		# valene value
			e = np.array([a,v])
			emotions_ = np.concatenate((emotions_, e), axis=0)
			# add current arousal and valence values to 16 timesteps (when values extracted from
			# video, it is done by default at a rate of 16 timesteps)
			for i in range(16):
				x_[0][j] = a
				x_[0][j+1] = v
				j += 158

			# every four lines of emotion values, send x_ into the gibbs sampling function
			# 4 lines = 64 timesteps -> size of visible layer = 64*158
			if l%4==3:
				# sample by running Gibbs chain 10 times
				sample = gibbs_sample_generate(10).eval(session=sess, feed_dict={x: x_, emotions: emotions_})

				# reshape the vector to be timesteps x notes (64x158), and then keep adding into one final matrix (song) 
				if 'song' in locals():
					S = np.reshape(sample, (num_timesteps, (2*note_range+2)))
					song = np.concatenate((song, S), axis=0)
				else:
					song = np.reshape(sample, (num_timesteps, (2*note_range+2)))
				
				# matrix returned from sample is the input into the next segment of 64 timesteps
				x_ = np.zeros((1, n_visible))
				# emotions_ and j are reset for next iteration
				emotions_ = np.array([])
				j=156

		# save the matrix as a midi file
		midi_manipulation.noteStateMatrixToMidi(song, midi_file_name)
	return

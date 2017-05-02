# Code based on Dan Shiebler's RBM music generator: https://github.com/dshieble/Music_RBM

import tensorflow as tf
import midi_manipulation

note_range = midi_manipulation.span # The range of notes that we can produce (=78)

num_timesteps  = 64 # Number of timesteps that we will create at a time (~= 4 measures/16 quarter notes)
n_visible      = (2*(note_range)+2)*(num_timesteps) # Number of nodes in the visible layer (=10112)
n_hidden       = 50 # Number of nodes in the hidden layer

num_epochs = 200 # Number of training epochs - each epoch we go through the entire data set
batch_size = 100 # Number of training examples that are sent through the RBM at a time
lr         = tf.constant(0.005, tf.float32) # The learning rate of our model

def get_variables():
	"""
	Returns variables used to train the model and generate from the model

	:param x: placeholder that holds music data
	:param emotions: placeholder that holds the emotion data
	:param W: matrix that stores the edge weights between the visible and hidden layers
	:param bh: vector that stores the bias values for the hidden layer
	:param bv: vector that stores the bias values for the visible layer
	:type x: tensor
	:type emotions: tensor
	:type W: tensor
	:type bh: tensor
	:type bv: tensor
	:returns: all variables defined (x, emotions, W, bh, bv)
	rtype: tuple of tensors
	"""
	x  = tf.placeholder(tf.float32, [1, n_visible], name="x") 
	emotions  = tf.placeholder(tf.float32, [8,], name="emotions") 
	W  = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W") 
	bh = tf.Variable(tf.zeros([1, n_hidden],  tf.float32, name="bh")) 
	bv = tf.Variable(tf.zeros([1, n_visible],  tf.float32, name="bv")) 
	return x, emotions, W, bh, bv


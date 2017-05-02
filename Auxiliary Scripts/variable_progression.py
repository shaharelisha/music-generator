import glob
import tensorflow as tf
import rbm
import re
import matplotlib.pyplot as plt


def variable_progression():
	"""
	Generate a list for the progression of each variable (W, bh, bv), by finding the values saved
	in each epoch checkpoint file
	"""
	# files = glob.glob('parameter_checkpoints/epoch-*[!.meta]')
	files = glob.glob('parameter_checkpoints/epoch-*')

	# reorder epochs by 'human order' otherwise it would order it as 1,110,12,...
	# http://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
	def atoi(text):
	    return int(text) if text.isdigit() else text

	def natural_keys(text):
	    '''
	    alist.sort(key=natural_keys) sorts in human order
	    http://nedbatchelder.com/blog/200712/human_sorting.html
	    (See Toothy's implementation in the comments)
	    '''
	    return [ atoi(c) for c in re.split('(\d+)', text) ]

	files.sort(key=natural_keys)

	x, W, bh, bv = rbm.get_variables()
	trainable_vars = [W, bh, bv]

	saver = tf.train.Saver(trainable_vars)	# restore the weights and biases of the trained model

	weights = []
	bhs = []
	bvs = []
	with tf.Session() as sess:
		init = tf.initialize_all_variables()	
		sess.run(init)
		# iterate through each saved epoch checkpoint, and add the W, bh, and bv matrices to their
		# respective lists
		for f in files:
			saver.restore(sess, f)		# load the saved weights and biases from a given epoch checkpoint file
			weights.append(W.eval())	
			bhs.append(bh.eval())
			bvs.append(bv.eval())

	return weights, bhs, bvs


def plot_progression(weights, bhs, bvs):
	"""
	Given lists of the weights and biases matrices as they progress over time during the training process, plot
	the variable progression over time. This version specifically plots the weights, but it can easily be changed
	to any of other variables 
	"""
	weights_plot = []
	for i in range(40):
		weights_plot.append(weights[i][0][0])	# only plots the first value in the matrix every time
	plt.plot(weights_plot)

	plt.show()


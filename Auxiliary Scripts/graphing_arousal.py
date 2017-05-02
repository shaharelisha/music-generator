import numpy as np
import matplotlib.pyplot as plt

# create array of numbers from 0 to 1 (inclusive) at intervals of 0.01
x = np.arange(0., 1.01, 0.01)


x_ = x*0.85+0.01				# transform x values
graph  = abs(np.log(x_))		# get absolute values of the natural log of x

plt.plot(x, graph, 'b-')	# plot points

plt.show()
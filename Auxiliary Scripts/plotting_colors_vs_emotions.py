from dominant_colors import run_dominant_colors
from improved_emotions_api import run_emotion_api

import matplotlib.pyplot as plt
import numpy as np

_key = '814b02959abc476fb96ed02f0f4e3562'


def combined_emotions(emotion_api_list, color_list):
	"""
	Returns a new list of averaged arousal and valence values, given the color analysis and the Emotion API. 
	"""
	if emotion_api_list != []:
		# loop and replace all ('n', 'n') values with the values of the color_list for that timestep until either
		# there are no more ('n', 'n') values or the color_list ends first
		while True:
			try:
				i = emotion_api_list.index(('n\n','n\n'))
			except ValueError: 
				break
			try:
				emotion_api_list[i] = color_list[i]
			except IndexError:
				break 							# the rest of the list won't be used anyways when the two lists are zipped below

		averaged_emotions_list = [((a_color+float(a_emotion))/2, (v_color+float(v_emotion))/2) for (a_color,v_color),(a_emotion, v_emotion) in zip(color_list, emotion_api_list)]
		# add remaining color_list values at the end if color_list is larger than emotion_api_list
		# usually the case if there are no faces at the end of the video
		if len(emotion_api_list) < len(color_list):
			emotion_api_list_length = len(emotion_api_list)
			averaged_emotions_list += color_list[emotion_api_list_length:]

	else:
		averaged_emotions_list = color_list
	return averaged_emotions_list


def save_colors(file_name):
	"""
	Given a video file name, extract emotions using color analysis and save the arousal and valence values
	in separate text files
	"""
	color_list = run_dominant_colors(file_name, multiple=16)
	file_name = file_name.split('.')[0]
	fa = open('%s-colors-arousal.txt' %file_name, 'w+')
	fv = open('%s-colors-valence.txt' %file_name, 'w+')
	for (a,v) in color_list:
		fa.write("%f\n" %a)
		fv.write("%f\n" %v)
	fa.close()
	fv.close()
	return None


def save_api_emotions(file_name):
	"""
	Given a video file name, extract emotions using the Emotion API and save the arousal and valence values
	in separate text files
	"""
	_key = '814b02959abc476fb96ed02f0f4e3562'
	emotion_api_list = run_emotion_api(file_name, _key=_key, multiple=16)
	file_name = file_name.split('.')[0]
	fa = open('%s-api-arousal.txt' %file_name, 'w+')
	fv = open('%s-api-valence.txt' %file_name, 'w+')
	for (a,v) in emotion_api_list:
		fa.write("%s\n" %a)
		fv.write("%s\n" %v)
	fa.close()
	fv.close()
	return None


def save_combined_emotions(file_name):
	"""
	Given a video file name, calculate the combined, averaged arousal and valence values by loading 
	the text files holding the emotion values from the color analysis and Emotion API calculations. 
	Save the averaged arousal and valence values in separate text files
	"""
	file_name = file_name.split('.')[0]

	color_list = []
	color_arousal_file = file_name + '-colors-arousal.txt'
	color_valence_file = file_name + '-colors-valence.txt'
	c_a = open(color_arousal_file, 'r')
	c_v = open(color_valence_file, 'r')
	for a, v in zip(c_a, c_v):
		color_list.append((float(a), float(v)))

	emotion_api_list = []
	api_arousal_file = file_name + '-api-arousal.txt'
	api_valence_file = file_name + '-api-valence.txt'
	a_a = open(api_arousal_file, 'r')
	a_v = open(api_valence_file, 'r')
	for a, v in zip(a_a, a_v):
		emotion_api_list.append((a, v))

	combined_list = combined_emotions(emotion_api_list, color_list)

	fa = open('%s-arousal.txt' %file_name, 'w+')
	fv = open('%s-valence.txt' %file_name, 'w+')
	for (a,v) in combined_list:
		fa.write("%f\n" %a)
		fv.write("%f\n" %v)
	fa.close()
	fv.close()
	return None


def plot_arousal_values(file_name):
	"""
	Given a video file name, load the following three text files: arousal values from color analysis, from Emotion API, 
	and from the combined, averaged calculations. Add all values into three lists respectively. Each list corresponds to
	a line; plot all lines on the same graph.
	"""

	file_name = file_name.split('.')[0]

	color_list_arousal = []
	color_arousal_file = file_name + '-colors-arousal.txt'
	c_a = open(color_arousal_file, 'r')
	for a in c_a:
		color_list_arousal.append(float(a))

	emotion_api_list_arousal = []
	api_arousal_file = file_name + '-api-arousal.txt'
	a_a = open(api_arousal_file, 'r')
	for a in a_a:
		if a != 'n\n':
			emotion_api_list_arousal.append(float(a))
		else:
			emotion_api_list_arousal.append(None)

	combined_list_arousal = []
	combined_arousal_file = file_name + '-arousal.txt'
	c_a = open(combined_arousal_file, 'r')
	for a in c_a:
		combined_list_arousal.append(float(a))

	purple = '#c275ce'
	green = '#78ba89'
	orange = '#f7ca4f'
	fig, ax = plt.subplots()
	ax.plot(emotion_api_list_arousal, color=purple, linestyle='-', marker='.', label="Emotion API")
	ax.plot(color_list_arousal, color=green, linestyle='-', marker='.', label="Color Analysis")
	ax.plot(combined_list_arousal, color=orange, label="Combined")

	# http://matplotlib.org/1.3.0/examples/pylab_examples/legend_demo.html
	legend = ax.legend(loc='upper right')

	for label in legend.get_texts():
		label.set_fontsize('large')

	for label in legend.get_lines():
		label.set_linewidth(1.5)

	plt.ylim([0,1])
	plt.ylabel('Value')
	plt.xlabel('Steps')
	plt.title('Arousal Values - ' + file_name)
	plt.show()


def plot_valence_values(file_name):
	"""
	Given a video file name, load the following three text files: valence values from color analysis, from Emotion API, 
	and from the combined, averaged calculations. Add all values into three lists respectively. Each list corresponds to
	a line; plot all lines on the same graph.
	"""
	file_name = file_name.split('.')[0]

	color_list_valence = []
	color_valence_file = file_name + '-colors-valence.txt'
	c_v = open(color_valence_file, 'r')
	for v in c_v:
		color_list_valence.append(float(v))

	emotion_api_list_valence = []
	api_valence_file = file_name + '-api-valence.txt'
	a_v = open(api_valence_file, 'r')
	for v in a_v:
		if v != 'n\n':
			emotion_api_list_valence.append(float(v))
		else:
			emotion_api_list_valence.append(None)

	combined_list_valence = []
	combined_valence_file = file_name + '-valence.txt'
	c_v = open(combined_valence_file, 'r')
	for v in c_v:
		combined_list_valence.append(float(v))

	purple = '#c275ce'
	green = '#78ba89'
	orange = '#f7ca4f'
	fig, ax = plt.subplots()
	ax.plot(emotion_api_list_valence, color=purple, linestyle='-', marker='.', label="Emotion API")
	ax.plot(color_list_valence, color=green, linestyle='-', marker='.', label="Color Analysis")
	ax.plot(combined_list_valence, color=orange, label="Combined")

	# http://matplotlib.org/1.3.0/examples/pylab_examples/legend_demo.html
	legend = ax.legend(loc='upper right')

	for label in legend.get_texts():
		label.set_fontsize('large')

	for label in legend.get_lines():
		label.set_linewidth(1.5)

	plt.ylim([0,1])
	plt.ylabel('Value')
	plt.xlabel('Steps')
	plt.title('Valence Values - ' + file_name)
	plt.show()


# def plot_color_list(file_name):
	# """
	# Given a video file name, a list of arousal and valence values extracted from the color analysis method
	# is then used to plot the arousal and valence values on the same graph
	# """
# 	color_list = run_dominant_colors(file_name, multiple=16)
# 	arousal = []
# 	valence = []
# 	i=0
# 	for a,v in color_list:
# 		arousal.append(a)
# 		valence.append(v)
# 		i += 1

# 	fig, ax = plt.subplots()
# 	ax.plot(arousal, 'r', label="Arousal")
# 	ax.plot(valence, 'b', label="Valence")

# 	plt.axis([0, i, 0, 1])
# 	plt.ylabel('Value')
# 	plt.xlabel('Steps')
# 	plt.show()


def plot_emotions_for_video(text_file):
	"""
	Given a text file listing the arousal and valence values for a video, this method plots 
	both lines on the same graph.
	"""
	text = open(text_file, 'r')

	arousal = []
	valence = []
	i = 0
	for line in text:
		emotion = line[:-1].split(",")

		arousal.append(float(emotion[0]))
		valence.append(float(emotion[1]))

		i += 1

	fig, ax = plt.subplots()
	ax.plot(arousal, 'r', label="Arousal")
	ax.plot(valence, 'b', label="Valence")

	# http://matplotlib.org/1.3.0/examples/pylab_examples/legend_demo.html
	legend = ax.legend(loc='upper right')

	for label in legend.get_texts():
		label.set_fontsize('large')

	for label in legend.get_lines():
		label.set_linewidth(1.5)

	plt.axis([0, i, 0, 1])
	plt.ylabel('Value')
	plt.xlabel('Steps')
	plt.show()

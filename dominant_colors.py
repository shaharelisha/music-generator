from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import numpy as np
import itertools
from math import factorial
from tqdm import tqdm
import moviepy.editor as mp


def centroid_histogram(clt):
	"""
	Given the centroids for a frame, the histogram returns the frequency of each centroid (color). 
	Method taken from:
	http://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/

	:param clt: clusters
	:type clt: sklearn.cluster.k_means_.KMeans
	:returns: an array of how frequent each dominant color is
	:rtype: numpy.ndarray
	"""
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins=numLabels)

	hist = hist.astype("float")
	hist /= hist.sum()

	return hist


def color_to_valence(color):
	"""
	Translates a color from HSV format to a valence value. Valence value directly correlates to both the saturation
	and value of the color. The average of the two is taken, and then normalised to a value between 0 and 1. 

	:param color: color in HSV format
	:type color: numpy.ndarray
	:returns: valence value from color
	:rtype: float
	"""
	# values are between 0-255
	s = float(color[1])			
	v = float(color[2])			

	average_valence = (s+v)/2
	normalised_valence = average_valence/255
	return normalised_valence


def travel_to_point(value, center_point, pure_point):
	"""
	Travels a percentage (value) of the way from a given center point to a given destination (pure) point.

	:param value: value between 0-1 determining how far to move towards a destination point
	:param center_point: a starting point
	:param pure_point: destination point
	:type value: float
	:type center_point: numpy.ndarray
	:type pure_point: numpy.ndarray
	:returns: vector representing a percentage (value) of the distance between the center_point and the pure_point
	:rtype: numpy.ndarray
	"""
	vector = pure_point - center_point
	final_vector = vector * value
	return final_vector


def shortest_distance((x,y)):
	"""
	Calculates the shortest distance between two numbers in a cycle of 0 to 180 (hue values). For example, 
	the distance between 20 and 170 is 30. 

	:param x: first number of the pair
	:param y: second number of the pair
	:type x: numpy.uint8
	:type y: numpy.uint8
	:returns: the minimum of the absolute value of the difference between x and y and 180 minus the absolute value 
	of the difference between the two numbers
	:rtype: int
	"""
	x = int(x)
	y = int(y)
	return min(abs(x-y), 180-abs(x-y))


def calculate_arousal_from_color_diversity(color_list, max_distance):
	"""
	Calculates an arousal value based on the diversity of the colors (hues) in a given frame. The higher the variety in color,
	the higher the arousal value. The value is then normalised to be between 0 and 1 by dividing by the
	maximum distance possible.

	:param color_list: list of (color, percentage) values for a given frame
	:param max_distance: maximum distance possible (when all points are equidistant)
	:type color_list: list
	:type max_distance: float
	:returns: an arousal value based on the variety/diversity of colors
	:rtype: float
	"""
	total_distance = 0
	hue_list = []
	for color, _ in color_list:
		hue_list.append(color[0])
	# sums the shortest distances for all unique combinations of hue pairs  
	for pair in itertools.combinations(hue_list, 2):
		total_distance += shortest_distance(pair)

	# TODO: exception handling if return is larger than 1
	normalised_arousal = float(total_distance)/max_distance				# normalise so value is between 0 and 1
	return normalised_arousal


def averaged_emotion(color_list, k):
	"""
	Calculates the maximum distance possible (when k amount of points are equidistant on a 
	180 point cycle), then returns a tuple of the average arousal and valence values for a given frame. 

	:param color_list: list of (color, percentage) values for a given frame
	:param k: number of clusters (equivalent to number of color values in color_list)
	:type color_list: list
	:type k: int
	:returns: a tuple of (arousal, valence)
	:rtype: tuple
	"""
	center_point = np.array([0.5,0.5], dtype=np.float)[np.newaxis].T
	final_point = center_point

	# find maximum distance given all points are equidistant from one another
	equidistance = float(180)/k
	equidistant_list = []
	for x in range(k):
		equidistant_list.append(x*equidistance)

	max_distance = 0
	for max_pair in itertools.combinations(equidistant_list, 2):
		max_distance += shortest_distance(max_pair)

	# calculate arousal and valence values
	arousal = calculate_arousal_from_color_diversity(color_list, max_distance)
	for color,percent in color_list:
		valence = color_to_valence(color)
		point = np.array([arousal,valence], dtype=np.float)[np.newaxis].T
		color_point_vector = travel_to_point(percent, center_point, point)
		final_point += color_point_vector

	arousal = final_point[0][0]
	valence = final_point[1][0]
	return arousal, valence


def run_dominant_colors(file_name, music_bpm=120, k=8, multiple=16):
	"""
	Iterates through every x frames of the videos, where x is the amount of frames per timestep of the 
	corresponding MIDI file. For every frame, k amount of the most dominant colors (and their percentage of 
	dominance) are extracted using the k-means clustering algorithm. Using these dominant colors, the average
	arousal and valence values are calculated for that frame. These emotion values per timestep (or multiple 
	timesteps) are added to the returned list.

	:param file_name: name of the video file
	:param music_bpm: beats per minute (bpm) of the song corresponding to the video, by default 120
	:param k: number of clusters for k-means clustering algorithm, by default 8
	:param multiple: allows for an average (arousal, valence) value for the given amount of timesteps
	:type file_name: str
	:type music_bpm: int
	:type k: int
	:type multiple: int
	:returns: list of averaged arousal, valence values for every timestep of the corresponding song
	:rtype: list
	"""
	
	cap = cv2.VideoCapture(file_name)
	ret = True
	fps = cap.get(cv2.cv.CV_CAP_PROP_FPS) 		# gets the frames per second of the video
	fps = round(fps)
	num_frames_per_timestep = ((fps*15)/music_bpm)*multiple
	length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) 		# gets total frame count, used for progress meter
	emotions_list_per_timestep = []
	with tqdm(total=length) as pbar:
		while ret:
			pbar.update(1)
			frame_num = int(round(cap.get(1))) 		# current frame number, rounded because frame intervals aren't always integers
			ret, frame = cap.read()
			if frame is None:
				break

			if frame_num % num_frames_per_timestep == 0.0:
				frame = cv2.resize(frame, (480,360))
				color_list = []
				image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)					# convert to HSV color space
				image = image.reshape((image.shape[0] * image.shape[1], 3))
				# http://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
				clt = KMeans(n_clusters=k)
				clt.fit(image)
				hist = centroid_histogram(clt)
				centroids = clt.cluster_centers_

				for color_pair in zip(centroids, hist):	
					color_list.append(color_pair)

				emotions_list_per_timestep.append(averaged_emotion(color_list, k))
	pbar.close()
	cap.release()
	cv2.destroyAllWindows()
	return emotions_list_per_timestep


from dominant_colors import run_dominant_colors
from emotions_api import run_emotion_api


def video_to_emotion(file_name, music_bpm=120, k=8, multiple=16):
	"""
	Returns a new list of averaged arousal and valence values, calculated from both the
	color analysis and the Emotion API. 

	:param file_name: name of the video file
	:param music_bpm: beats per minute (bpm) of the song corresponding to the video, by default 120
	:param k: number of clusters for k-means clustering algorithm, by default 8
	:param multiple: allows for an average (arousal, valence) value for the given amount of timesteps, default 16
	:type file_name: str
	:type music_bpm: int
	:type k: int
	:type multiple: int
	:returns: list of averaged arousal, valence values for every timestep of the corresponding song
	:rtype: list
	"""
	_key = # add your Emotion API key here

	color_list = run_dominant_colors(file_name, music_bpm, k, multiple)
	emotion_api_list = run_emotion_api(file_name, _key, music_bpm, multiple)

	# checks if the Emotion API returned any data
	if emotion_api_list != []:

		# loop and replace all ('n', 'n') values with the values of the color_list for that timestep until either
		# there are no more ('n', 'n') values or the color_list ends first
		while True:
			try:
				i = emotion_api_list.index(('n','n'))
			except ValueError: 
				break
			try:
				emotion_api_list[i] = color_list[i]
			except IndexError:
				break 							# the rest of the list won't be used anyways when the two lists are zipped below

		averaged_emotions_list = [((a_color+a_emotion)/2, (v_color+v_emotion)/2) for (a_color,v_color),(a_emotion, v_emotion) in zip(color_list, emotion_api_list)]
		# add remaining color_list values at the end if color_list is larger than emotion_api_list
		# usually the case if there are no faces at the end of the video
		if len(emotion_api_list) < len(color_list):
			emotion_api_list_length = len(emotion_api_list)
			averaged_emotions_list += color_list[emotion_api_list_length:]

	else:
		averaged_emotions_list = color_list
	return averaged_emotions_list


def video_to_emotion_as_file(file_name, music_bpm=120, k=8, multiple=16):
	"""
	Creates a text file with the same name as the video, calculates all the arousal and valence values 
	for the video, and writes them to the text file. Used in order to save time when training the network.

	:param file_name: name of the video file
	:param music_bpm: beats per minute (bpm) of the song corresponding to the video, by default 120
	:param k: number of clusters for k-means clustering algorithm, by default 8
	:param multiple: allows for an average (arousal, valence) value for the given amount of timesteps, default 16
	:type file_name: str
	:type music_bpm: int
	:type k: int
	:type multiple: int
	:returns: None
	:rtype: None
	"""
	emotions_list = video_to_emotion(file_name, music_bpm, k, multiple)
	file_name = file_name.split('.')[0]
	f = open('%s.txt' %file_name, 'w+')
	for (a,v) in emotions_list:
		f.write("%f, %f\n" %(a,v))
	f.close()
	return None



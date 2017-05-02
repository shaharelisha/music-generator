from midiutil.MidiFile import MIDIFile
import math
import numpy as np

def create_arpeggios_chord_variation(emotion_text_file, file_name):
	"""
	Given the emotion values from a video, creates a midi file where the notes are dependent on 
	the emotion values. The higher the arousal value, the faster the arpeggio is played. The higher
	the valence value, the more major the chord played.

	:param emotion_text_file: path to text file holding emotion values for video
	:param file_name: name for midi file_name
	:type emotion_text_file: str
	:type file_name: str
	:returns: None
	:rtype: None 
	"""
	MyMIDI = MIDIFile(1)

	track = 0 
	time = 0

	# creates a template for a midi track
	MyMIDI.addTrackName(track,time, file_name) 
	MyMIDI.addTempo(track,time,120)		# 120 bpm

	channel = 0 
	volume = 120
	base_note = 60

	# reads arousal and valence values from emotions text file and adds them to an arousal
	# list and a valence list respectively
	arousal_list = []
	valence_list = []
	with open(emotion_text_file, 'r') as emotions_text:
		for emotion in emotions_text:
			emotion = emotion[:-1].split(",")
			arousal_list.append(float(emotion[0]))
			valence_list.append(float(emotion[1]))

	# finds range of valence values specific to the video
	min_v = min(valence_list)
	max_v = max(valence_list)

	# creates 12 boundaries ("thresholds") of values within range
	steps = (max_v-min_v)/4
	thresholds = np.arange(min_v, max_v+steps, steps)

	duration = arousal_list[0]

	# for arousal, valence in emotions list:
	for arousal, valence in zip(arousal_list, valence_list):
		# arousal to duration relationship is logarithmic - difference when arousal is low
		# is more evident, so arousal values are slightly shifted as to never be 1 or 0
		arousal = arousal * 0.85 + 0.01
		arousal = abs(math.log(arousal))

		# determine chord/notes of arpeggio that will be played, depending on valence value
		if thresholds[0] <= valence < thresholds[1]:
			i = 3
		elif thresholds[1] <= valence < thresholds[2]:
			i = 2
		elif thresholds[2] <= valence < thresholds[3]:
			i = 5
		else:
			i = 4

		# add notes until total duration reaches 6
		total_length = 0
		under_limit = True
		while under_limit:
			duration = arousal
			for pitch in [base_note, base_note+i, base_note+7, base_note+i]:
				MyMIDI.addNote(track,channel,pitch,time,duration,volume)	# add note to midi track
				total_length += duration
				time += duration
				if total_length > 6:		
					under_limit = False
					break

	# write midi file
	file_name = file_name + '.mid'
	binfile = open(file_name, 'wb') 
	MyMIDI.writeFile(binfile) 
	binfile.close()
	return
import midi_manipulation
import numpy as np
import matplotlib.pyplot as plt


def get_histogram(midi_file):
	"""
	Given a midi file, convert the midi file into state matrix and add all matrices to get the frequency
	of each note in the song
	"""
	song = midi_manipulation.midiToNoteStateMatrix(midi_file)
	song = np.array(song)

	x,y = song.shape
	total = np.zeros(y,)

	for i in range(x):
		total += song[i]

	return total


def get_notes(midi_file):
	"""
	Given a midi file, calculate a frequency of each note in the song. For each occurance of a note, add it to the list.
	Plot this list to get a histogram diagram

	A simplified example: [0,2,4,1] would equate to [1,1,2,2,3,3,3,3,4]
	"""
	total = get_histogram(midi_file)
	histogram = []
	notes = total[:78]
	for i in range(78):
		num = int(notes[i])
		for j in range(num):
			histogram.append(i+24)

	bins = np.linspace(midi_manipulation.lowerBound, midi_manipulation.upperBound, num=midi_manipulation.span+1)

	plt.hist(histogram, bins=bins)
	plt.xlim([24, 102])
	plt.title(midi_file)
	plt.ylabel("Frequency")
	plt.xlabel("Pitch (Midi)")
	plt.show()


def get_accuracy(midi_file, pitch):
	"""
	Calculate the accuracy by taking a midi file and calculating the frequency of the stated pitch compared to the total.
	Note that from the 'total' matrix, only the first 78 elements are taken. Those represent the notes, whereas the next 78
	represent the duration.
	"""
	total = get_histogram(midi_file)
	correct = total[pitch-24]
	overall = total[:78].sum()
	print overall
	print float(correct)/overall

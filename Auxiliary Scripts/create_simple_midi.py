from midiutil.MidiFile import MIDIFile

def create_midi(pitch):
	"""
	Creates a midi file of just one note repeated 32 times

	:param pitch: pitch of note
	:type pitch: int
	:returns: None
	:rtype: None 
	"""
	MyMIDI = MIDIFile(1)

	track = 0 
	time = 0

	# creates a template for a midi track
	MyMIDI.addTrackName(track,time, str(pitch)) 
	MyMIDI.addTempo(track,time,120)

	channel = 0 
	volume = 120
	# pitch = 60
	duration = 1

	for i in range(3200):
		MyMIDI.addNote(track,channel,pitch,time,duration,volume)
		time += duration

	# write midi file
	file_name = str(pitch) + '.mid'
	binfile = open(file_name, 'wb') 
	MyMIDI.writeFile(binfile) 
	binfile.close()
	return
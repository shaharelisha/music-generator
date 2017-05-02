import scipy.io.wavfile
import pretty_midi
from moviepy.editor import *

def overlay_music_and_video(midi_file, video_filename):
	"""
	Synthesizes midi file and overlays generated music onto original video clip

	:param midi_file: path to midi file
	:param video_filename: path to video file
	:type midi_file: str
	:type video_filename: str
	:returns: None
	:rtype: None
	"""
	# synthesize midi file using PrettyMidi
	midi_data = pretty_midi.PrettyMIDI(midi_file)
	audio_data = midi_data.synthesize()

	midi_file_name = midi_file.split('.')[0]

	# save synthesized midi file as wav file
	output_name = midi_file_name + '.wav'
	scipy.io.wavfile.write(output_name, 44100, audio_data)

	# overlay wav file and video file
	audioclip = AudioFileClip(output_name)
	videoclip = VideoFileClip(video_filename)
	duration = videoclip.duration
	# if audio file is longer than video file, set the duration of the audio
	# file to the duration of the video file
	if duration < audioclip.duration:
		videoclip2 = videoclip.set_audio(audioclip.set_duration(duration))
	else:
		videoclip2 = videoclip.set_audio(audioclip)

	# save as new video file
	names = video_filename.split('.')
	extension = names[1]
	video_filename = names[0]
	final_video = video_filename + '_with_music.mp4'

	videoclip2.write_videofile(final_video)

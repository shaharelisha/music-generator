import argparse
import sys
import combined_emotions
import overlaying_music
import chords
import scales


def main():
	"""
	Given a video path and a music generator option, the system will extract the emotions from the video, 
	generate music dependent of the emotions, and overlay the music onto the original video. This process
	will create 4 new files: a text file of the extract emotion values, a midi file of the music generated, 
	a wav file of the synthesized midi file, and an mp4 file of the original video with the generated music.

	This method is run through the command-line interface
	:param video_path: path to video file from current location
	:param generator_opt: music generator option: 1 (Scales), 2 (Chords), 3 (RBM)
	:type video_path: str
	:type generator_opt: int
	:returns: None
	:rtype: None
	"""
	parser = argparse.ArgumentParser()

	parser.add_argument("video_path", help="Path to video file from current location",
                    	type=str)

	parser.add_argument("generator_opt", help="Music generator option: 1 (Scales), 2 (Chords), 3 (RBM - Feedback), 4 (RBM - Sampling)",
                    	type=int)

	args = parser.parse_args()

	generator_option = args.generator_opt

	if generator_option in [1,2,3,4]:
		videofile = args.video_path
		file_name = videofile.split('.')[0]
		emotion_text_file = '%s.txt' %file_name

		if generator_option == 1:
			# Scale variant music generation option
			print "Analysing video and extracting emotion values"
			combined_emotions.video_to_emotion_as_file(videofile, multiple=24)
			print "Generating music!"
			scales.create_arpeggios_scale_variation(emotion_text_file, file_name)
		if generator_option == 2:
			# Chord variant music generation option
			print "Analysing video and extracting emotion values"
			combined_emotions.video_to_emotion_as_file(videofile, multiple=24)
			print "Generating music!"
			chords.create_arpeggios_chord_variation(emotion_text_file, file_name)
		if generator_option == 3:
			import generate_feedback
			# RBM (feedback) generated music
			print "Analysing video and extracting emotion values"
			combined_emotions.video_to_emotion_as_file(videofile)
			print "Generating music!"
			generate_feedback.generate_music(emotion_text_file, file_name)
		if generator_option == 4:
			import generate_sampling
			# RBM (sampling) generated music
			print "Analysing video and extracting emotion values"
			combined_emotions.video_to_emotion_as_file(videofile)
			print "Generating music!"
			generate_sampling.generate_music(emotion_text_file, file_name)
		
		print "Overlaying music generated with video"
		midi_file = file_name + '.mid'
		overlaying_music.overlay_music_and_video(midi_file, videofile)
	else:
		# if the music generator is not one of the pre-defined options, print the helper and exit 
		# the system
		parser.print_help()
		sys.exit()
		
if __name__ == "__main__":
    main()
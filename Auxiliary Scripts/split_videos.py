from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip

#https://www.reddit.com/r/moviepy/comments/2bsnrq/is_it_possible_to_get_the_length_of_a_video/
#http://stackoverflow.com/questions/37317140/cutting-out-a-portion-of-video-python

def split_video(videofile):
	"""
	Given a path to a video file, load the video and split the videos into 30 second segments. The last segment would be
	the rest of the video, even if it's less than 30 seconds. The video segments are saved with the original name followed
	by their number, in the same folder as the original video. 

	Example: input file = "example.mp4" -> "example1.mp4" "example2.mp4" "example3.mp4"
	"""
	clip = VideoFileClip(videofile)
	start = 0
	end = clip.duration
	i = 1
	filename = videofile.split('.')
	extension = filename[1]
	filename = filename[0]

	while start+30 <= end:
		target_name = filename + str(i) + '.' + extension
		ffmpeg_extract_subclip(videofile, start, start+30, targetname=target_name)
		start += 30
		i += 1

	target_name = filename + str(i) + '.' + extension
	ffmpeg_extract_subclip(videofile, start, end, targetname=target_name)
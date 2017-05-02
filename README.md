# Learning Cross-Modal Mappings for Media Generation

This repository contains code for generating music adaptive to video input. Four music-generation approaches have been implemented: two deterministic models and two stochastic. The deterministic models express a variety in chords, scales, and speed in response to a change in emotion values, whereas the stochastic models use an RBM learning method to generate music that corresponds to the visuals of a given video.


### Prerequisites

This whole project was developed on Ubuntu, using Python 2.7.


Instructions for installing TensorFlow on Ubuntu can be found here:
https://www.tensorflow.org/install/install_linux#InstallingVirtualenv

Ensure you have pip installed, then run the following command:

```
pip install -r requirements.txt
```

Create an account: https://www.microsoft.com/cognitive-services/en-us/emotion-api
Add your unique key to line 21 of combined_emotions.py

### Generating Music

Once the virtual environment has been set up and all the dependencies have been installed, clone the directory to start generating music. 

Add a video file into the directory, and pick one of the 4 music generating models:

Scales -> 1

Chords -> 2

RBM - Feedback -> 3

RBM - Sampling -> 4

Run the following command to generate music:

```
python generate_music.py <video_path> <generator_opt_number>
```

Once generation is finished, the following files would have been created:

* A new video file with the generated music
* A text file of the emotions extracted from the original video
* A MIDI file of the music generated
* A WAV file of the music generated


### Training RBM

If you want to train the system with your own dataset, populate a `Midi_Files` folder with a set of MIDI files and the corresponding videos' emotion values text files. 

To extract emotions from a given video, make sure you're in the same directory as the video and run the following code from the python shell:

```
import combined_emotions
combined_emotions.video_to_emotion_as_file(<video_file_name>)
```

Ensure that a `parameter_checkpoints` folder is in the directory.
Once the training dataset is all gathered, you can train the RBM by running the following code from the command line:

```
python training.py
```

Once training is complete, `parameter_checkpoints` should be populated with a series of `epoch_<x>.ckpt` files and a final checkpoint `trained_system.ckpt`

Running the music generating code will automatically use the trained system if `parameter_checkpoints` is within the same directory as `generate_music.py`


### Auxiliary Scripts

Used to plot various graphs and to split a given video into 30 second segments (faster to run emotion extraction on smaller videos)

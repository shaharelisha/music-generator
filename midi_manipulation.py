# Code based on Dan Shiebler's RBM music generator: https://github.com/dshieble/Music_RBM

import midi
import numpy as np
import glob
from tqdm import tqdm

lowerBound = 24
upperBound = 102
span = upperBound-lowerBound


def get_songs(path):
    """
    Iterates through all the midi files in a given folder, transforms each into a note state matrix, concatenates the
    emotion values at the end of each timestep, and adds the final matrix to a list of songs

    :param path: path to a folder containing midi files used for training the model
    :type path: str 
    :returns: list of all songs in matrix form
    :rtype: list
    """
    files = glob.glob('{}/*.mid*'.format(path))
    songs = []
    for f in tqdm(files):
        try:
            song = np.array(midiToNoteStateMatrix(f))
            x, y = song.shape
            emotion_array = np.zeros((x,2))
            i = 0
            text_file = f.replace("mid", "txt")

            # total number of emotion values in file. Following two lines of code taken from:
            with open(text_file) as f:
                total_lines = sum(1 for _ in f)

            repeats = x/total_lines   # used to determine how many timesteps are assigned the same emotion values

            text = open(text_file, 'r')     # required to reopen text file
            for line in text:
                emotion = line[:-1].split(",")
                a = float(emotion[0])
                v = float(emotion[1])
                for j in range(repeats):
                    emotion_array[i+j] = [a,v]
                i += repeats

            # add last emotion values to remaining timesteps
            j += 1
            i -= repeats
            while i+j < x:
                emotion_array[i+j] = [a,v]
                j += 1

            # add emotion values columns to the end of the song matrix
            song = np.hstack((song, emotion_array))

            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e           
    return songs


def midiToNoteStateMatrix(midifile, span=span):
    """
    Reads a midi file, and transforms it into a note state matrix of size 2*span.
    First half represents the notes currently playing, second half represents the whether
    the previous was held or not.

    :param midifile: path to midifile
    :param span: pitch range, default span (78)
    :type midifile: str
    :type span: int
    :returns: matrix representation of song
    :rtype: numpy array
    """
    pattern = midi.read_midifile(midifile)

    timeleft = [track[0].tick for track in pattern]

    posns = [0 for track in pattern]

    statematrix = []
    time = 0

    state = [[0,0] for x in range(span)]
    statematrix.append(state)
    condition = True
    while condition:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0],0] for x in range(span)]
            statematrix.append(state)
        for i in range(len(timeleft)): # For each track
            if not condition:
                break
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        pass
                    else:
                        if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch-lowerBound] = [0, 0]
                        else:
                            state[evt.pitch-lowerBound] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    if evt.numerator not in (2, 4):
                        # Ignore non-4 time signatures
                        out =  statematrix
                        condition = False
                        break
                try:
                    timeleft[i] = track[pos + 1].tick
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    S = np.array(statematrix)
    statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
    statematrix = np.asarray(statematrix).tolist()
    return statematrix


def noteStateMatrixToMidi(statematrix, name="example", span=span):
    """
    Transforms a matrix representation of notes and into a midi file.
    
    :param statematrix: matrix representation of song
    :param name: name for midi file
    :param span: pitch range, default span (78)
    :type statematrix: numpy array
    :type name: str
    :type span: int
    :returns: None
    :rtype: None
    """
    statematrix = np.array(statematrix)
    if not len(statematrix.shape) == 3:
        statematrix = np.dstack((statematrix[:, :span], statematrix[:, span:-2]))
    statematrix = np.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    
    span = upperBound-lowerBound
    tickscale = 55
    
    lastcmdtime = 0
    prevstate = [[0,0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):  
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=120, pitch=note+lowerBound))
            lastcmdtime = time
            
        prevstate = state
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)
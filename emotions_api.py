import httplib
import urllib
import base64
import json
import pandas as pd
import numpy as np
import requests
import time

def list_aggregate_emotion_per_timestep(rawData, num_frames_per_timestep):
    """
    Given the raw data of emotion values per frame for the whole video, the emotions are averaged for every 
    num_frames_per_timestep frames. The average emotions are then normalised to a sum of 1, and appended to 
    a list.

    :param rawData: processed results of the emotions from video by the Emotion API
    :param num_frames_per_timestep: amount of frames per timestep
    :type rawData: dict
    :type num_frames_per_timestep: int
    :returns: a list of the normalised emotions per timestep
    :rtype: list
    """
    list_of_emotions = []
    interval_length = 0

    # find interval length between frames
    try:
        for fragment in rawData['fragments']:
            try:
                interval_length = fragment['interval']
                break
            except KeyError:
                continue
    except KeyError:
        pass

    if interval_length == 0:
        # no face events detected
        return [0]
    else:
        # events exist - faces detected
        continuous_list = []
        for fragment in rawData['fragments']:
            duration = fragment['duration']
            try: 
                events = fragment['events']
                for e in events:
                    # empty frames within an event
                    if e == []:
                        emotions = {'neutral': 0.0, 'happiness': 0.0, 'surprise': 0.0, 'sadness': 0.0, 'anger': 0.0, 'fear': 0.0}
                        continuous_list.append(emotions)
                    # frames with faces and returned values
                    else:
                        interval = e[0]['scores']
                        neutral = float(interval['neutral'])
                        happiness = float(interval['happiness'])
                        surprise = float(interval['surprise'])
                        sadness = float(interval['sadness'])
                        anger = float(interval['anger'])
                        fear = float(interval['fear'])
                        emotions = {'neutral': neutral, 'happiness': happiness, 'surprise': surprise, 'sadness': sadness, 'anger': anger, 'fear': fear}
                        continuous_list.append(emotions)
            # no events in this fragment
            except KeyError:
                reps = duration / interval_length       # add this many empty frames (/emotions) to the list
                emotions = {'neutral': 0.0, 'happiness': 0.0, 'surprise': 0.0, 'sadness': 0.0, 'anger': 0.0, 'fear': 0.0}
                for i in range(reps):
                    continuous_list.append(emotions)
        i = 0
        while i < len(continuous_list):
            # average emotions every num_frames_per_timestep frames
            emotions_list = continuous_list[i:i+num_frames_per_timestep]
            neutral = 0.0
            happiness = 0.0
            surprise = 0.0
            sadness = 0.0
            anger = 0.0
            fear = 0.0
            for e in emotions_list:
                neutral += e['neutral']
                happiness += e['happiness']
                surprise += e['surprise']
                sadness += e['sadness']
                anger += e['anger']
                fear += e['fear']
                step_emotions = {'neutral': neutral, 'happiness': happiness, 'surprise': surprise, 'sadness': sadness, 'anger': anger, 'fear': fear}
            # sum for each value in the emotion dictionary
            sum_emotions = sum(step_emotions.itervalues())
            if sum_emotions == 0.0:
                list_of_emotions.append(('n','n'))
            else:
                # normalised to a sum of 1
                for k,v in step_emotions.iteritems():
                    step_emotions[k] = v/sum_emotions
                list_of_emotions.append(step_emotions)
            i += num_frames_per_timestep
    return list_of_emotions


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


def emotions_to_arousal_valence(emotions):
    """
    Translates and maps the different emotion values for a given timestep to a point on
    the arousal-valence diagram and returns the point as a tuple.

    :param emotions: dictionary of normalised values per emotion
    :type emotions: dict
    :returns: arousal, valence tuple
    :rtype: tuple
    """
    num_emotions = 6
    pure_neutral = np.array([0.5,0.5], dtype=np.float)[np.newaxis].T
    pure_happiness = np.array([1,1], dtype=np.float)[np.newaxis].T
    pure_surprise = np.array([1,0.5], dtype=np.float)[np.newaxis].T
    pure_sadness = np.array([0,0], dtype=np.float)[np.newaxis].T
    pure_anger = np.array([1,0], dtype=np.float)[np.newaxis].T
    pure_fear = np.array([1,0], dtype=np.float)[np.newaxis].T
    point_sum = pure_neutral + pure_happiness + pure_surprise + pure_sadness + pure_anger + pure_fear
    center_point = point_sum/num_emotions

    try:
        neutral_vector = travel_to_point(emotions['neutral'], center_point, pure_neutral)
        happiness_vector = travel_to_point(emotions['happiness'], center_point, pure_happiness)
        surprise_vector = travel_to_point(emotions['surprise'], center_point, pure_surprise)
        sadness_vector = travel_to_point(emotions['sadness'], center_point, pure_sadness)
        anger_vector = travel_to_point(emotions['anger'], center_point, pure_anger)
        fear_vector = travel_to_point(emotions['fear'], center_point, pure_fear)

        final_point = center_point + neutral_vector + happiness_vector + surprise_vector + sadness_vector + anger_vector + fear_vector
        
        arousal = final_point[0][0]
        valence = final_point[1][0]
    except KeyError:
        return ('n', 'n')
    return arousal, valence


def run(rawData, num_frames_per_timestep):
    """
    Given the processed results from the Emotion API for a given video, every set of emotions is 
    translated into a tuple of arousal, valence per timestep (or multiple timesteps), and appended to 
    a list of all the tuples. If an element of the emotions_list is an integer x instead of a dictionary
    of emotion values, that means that no data was processed (no face) for x amount of timesteps, 
    and x amount of tuples would be appended to the final list.

    :param rawData: processed results of the emotions from video by the Emotion API
    :param num_frames_per_timestep: amount of frames per timestep
    :type rawData: dict
    :type num_frames_per_timestep: int
    :returns: list of averaged arousal, valence values for every timestep of the corresponding song
    :rtype: list
    """
    emotions_list = list_aggregate_emotion_per_timestep(rawData, num_frames_per_timestep)
    arousal_valence_list = []
    for emotions in emotions_list:
        if type(emotions) == type(0) or type(emotions) == type(0.0):
            return []
        elif emotions == ('n', 'n'):
            arousal_valence_list.append(emotions)
        else:
            arousal, valence = emotions_to_arousal_valence(emotions) 
            arousal_valence_list.append((arousal, valence))
    return arousal_valence_list


def run_emotion_api(file_name, _key, music_bpm=120, multiple=16):
    """
    Calls the Emotion API and retrieves the processed data as emotion values per frame. Said values are
    translated into a list of arousal, valence value tuples per timestep (or multiple timesteps).

    :param file_name: name of the video file
    :param _key: key for Emotion API
    :param music_bpm: beats per minute (bpm) of the song corresponding to the video, by default 120
    :param multiple: allows for an average (arousal, valence) value for the given amount of timesteps
    :type file_name: str
    :type _key: str
    :type music_bpm: int
    :type multiple: int
    :returns: list of averaged arousal, valence values for every timestep of the corresponding song
    :rtype: list
    """
    _url = 'https://api.projectoxford.ai/emotion/v1.0/recognizeInVideo'
    _maxNumRetries = 10

    paramsPost = urllib.urlencode({'outputStyle' : 'perFrame', 'file': file_name})
    headersPost = dict()
    headersPost['Ocp-Apim-Subscription-Key'] = _key
    headersPost['content-type'] = 'application/octet-stream'

    try:
        responsePost = requests.request('post', _url + "?" + paramsPost, data = open(file_name,'rb').read(), \
                                        headers = headersPost)
    except IOError:
        print "File by the name provided not found"
        return

    if responsePost.status_code == 202:
        location = responsePost.headers['Operation-Location']

        ready = False
        while not ready:
            time.sleep(60)          # pause program for 60 seconds
            headersGet = dict()
            headersGet['Ocp-Apim-Subscription-Key'] = _key

            jsonGet = {}
            paramsGet = urllib.urlencode({})
            getResponse = requests.request('get', location, json = jsonGet, 
                data = None, headers = headersGet, params = paramsGet)
            try:
                rawData = json.loads(json.loads(getResponse.text)['processingResult'])
                fps = rawData['framerate']
                fps = round(fps)
                num_frames_per_timestep = ((fps*15)/music_bpm)*multiple
                ready = True
            except KeyError:
                continue
        return run(rawData, int(num_frames_per_timestep))
    else:
        print "unsuccessful"
        return []

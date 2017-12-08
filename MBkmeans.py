# MBkmeans.py
# Uses sci-kit MiniBatchKMeans library to cluster tracks
# by their lyrics which come in in bag-of-words format.

import hdf5_getters
import os
import csv
import string
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np 

# Provide the starting root dir 
# This is different for your particular setup! 
indir = '/Users/marianaecheverria/Desktop/cs451/finalproject/MillionSongSubset/data/'

def read_txt(filename):
    """ 
    Reads txt file and returns tuple with
    list of top 5,000 words and 
    the index : frequency for each track
    """
    content = [] # list with word index : word count for each track
    string = '%'
    find = False 
    words = [] 
    track_id = [] # list with track ID's from the MSD
    mxm_tid = [] # track ID's from musiXmatch
    str_data = []

    read_file = open(filename, "r")
    
    for line in read_file:
        if find:
            line = line.strip() # converting line into list
            index1 = line.find(',') # finds index of 1st comma
            index2 = line.find(',', index1+1) # finds index of 2nd comma
            track_id.append(line[:index1]) # appends track id to list 
            mxm_tid.append(line[:index2]) # appends track id to list 
            res = '{' + line[index2+1:] + '}' # simulates dictionary with string
            d = eval(res) # converts string to actual dictionary 
            content.append(d) # appends track data to content list
        else:
            # obtaining line with 5,000 words 
            if line.startswith(string):
                line = line[1:] # getting rid of %
                words = [word.strip() for word in line.split(',')]
                find = True # already found list of words 
    read_file.close() 
    
    return (words, content, track_id, mxm_tid)


def create_vectors(list_dict, num_words):
    """
    Returns a list x for all the data points. 
    
    Each element of x is a NumPy vector with 5,000 elements, 
    one for each word.
    """
    x = [] # list that will hold data 

    for d in list_dict:
        # initializing numpy vector
        # it contains 5,000 (number of words) zeros
        temp = np.zeros(num_words, dtype=np.float64)
        for key, val in d.items():
            key -= 1 # indexing in data starts at 1
            temp[key] = 1 # adding word and its frequency to vector 

        x.append(temp) # appends vector to x  

    return x


def create_csv(list_tracks, c):
    """
    Creates csv file with diffrent features of tracks.
    
    This csv file will be later used to
    predict the clusters to which each track was 
    asssigned to. 
    """
    # open the CSV file we will write to
    with open("output-lyrics-year.csv", 'wb') as csvfile:
        # Recursively visit each sub-dir till we reach the h5 files
        # Strip punctuation from features that are strings
        for root, dirs, filenames in os.walk(indir):
            for f in filenames:
                # gets track id
                track = f[:-3]
                # only add cluster labels to tracks that are found in 
                # root directory 
                if track in list_tracks:
                    # Use the hd5 wrappers to open the h5 file
                    h5 = hdf5_getters.open_h5_file_read(os.path.join(root, f))
                    # getting index of track in list tracks
                    ind = list_tracks.index(track)
                    # getting cluster of track
                    label = c[ind]

                    # EXTRACTING FEATURES 
                    # See hd5_getter.py for the various features that we can extract from each h5 file
                    # Get year
                    year = hdf5_getters.get_year(h5)
                    # Get artist HOTTTNESSSSSS
                    hotttness = hdf5_getters.get_artist_hotttnesss(h5)
                    # Get loudness
                    loudness = hdf5_getters.get_loudness(h5)
                    # Get tempo
                    tempo = hdf5_getters.get_tempo(h5)
                    # Get analysis sample rate
                    analysis_rate = hdf5_getters.get_analysis_sample_rate(h5)
                    # Get end of fade in 
                    end_of_fade_in = hdf5_getters.get_end_of_fade_in(h5)
                    # Get key
                    key = hdf5_getters.get_key(h5)
                    # Get key confidence
                    key_confidence = hdf5_getters.get_key_confidence(h5)
                    # Get mode
                    mode = hdf5_getters.get_mode(h5)
                    # Get mode confidence
                    mode_confidence = hdf5_getters.get_mode_confidence(h5)
                    # Get start of fade-out
                    start_of_fade_out = hdf5_getters.get_start_of_fade_out(h5)
                    # Get end of fade-in
                    end_of_fade_in = hdf5_getters.get_end_of_fade_in(h5)
                    # Get artist ID
                    artist_id = hdf5_getters.get_artist_7digitalid(h5)
                    # Get artist familiarity 
                    artist_familiarity = hdf5_getters.get_artist_familiarity(h5)
                    # Get time signature
                    time_signature = hdf5_getters.get_time_signature(h5)
                    # Get time signature confidence 
                    time_signature_conf = hdf5_getters.get_time_signature_confidence(h5)
                    # Close the h5 file
                    h5.close()

                    # Write to the CSV file
                    csvfile.write(str(label) + "," + str(year) + "," + str(hotttness) + "," + str(loudness) + \
                                    "," + str(year) + "," + str(tempo) + "," + str(analysis_rate) + \
                                    "," + str(end_of_fade_in) + "," + str(key) + \
                                    "," + str(key_confidence) + "," + str(mode) + "," + str(mode_confidence) +\
                                    "," + str(start_of_fade_out) + "," + str(end_of_fade_in) + "," +
                                    str(artist_id) + "," + str(artist_familiarity) + "," + str(time_signature) + \
                                    "," + str(time_signature_conf))
                    csvfile.write("\n")


if __name__ == "__main__":
    # Reading data from test txt
    (words, test_list, track_id1, mxm_tid1) = read_txt('mxm_dataset_test.txt')
    print('== Finished test txt')
    # Reading data from train txt
    (words, train_list, track_id2, mxm_tid2) = read_txt('mxm_dataset_train.txt')
    print('== Finished train txt')
    # creating vectors 
    x_test = create_vectors(test_list, 5000)
    x_train = create_vectors(train_list, 5000)
    X = np.concatenate((x_test, x_train), axis=0)
    tracks = track_id1 + track_id2

    print('== Beginning clustering')
    kmeans = MiniBatchKMeans(n_clusters=3, random_state=0).fit(X)
    cost = (kmeans.inertia_)/ len(X)
    mu = kmeans.cluster_centers_
    m = len(X)
    c = kmeans.labels_
    print('J = ' + str(kmeans.inertia_ / m))


    create_csv(tracks, c)

    dict_count = dict()

    # Prints how many tracks are assigned to each cluster 
    for el in c:
        if el in dict_count.keys():
            dict_count[el] += 1
        else:
            dict_count[el] = 1

    print(dict_count)


# 
# kmeans using sci-kit

from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np 
import csv
import random
from scipy.stats import mode
import matplotlib.pyplot as plt


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
    # maxval = 0
    # minval = 0
    # res = 0

    for d in list_dict:
        # initializing numpy vector
        # it contains 5,000 (number of words) zeros
        temp = np.zeros(num_words, dtype=np.float64)
        for key, val in d.items():
            if key < num_words:
                key -= 1 # indexing in data starts at 1
                temp[key] = 1 # adding word and its frequency to vector 
                # temp[key] = val
        x.append(temp) # appends vector to x  
    
    return np.array(x)



if __name__ == "__main__":
    (words, test_list, track_id, mxm_tid) = read_txt('mxm_dataset_test.txt')
    print('== Finished test txt')
    (words, train_list, track_id, mxm_tid) = read_txt('mxm_dataset_train.txt')
    print('== Finished train txt')
    #print(words)
    x_test = create_vectors(test_list, 5000)
    x_train = create_vectors(train_list, 5000)
    X = np.concatenate((x_test, x_train), axis=0)
    print('== Beginning clustering')
    kmeans = MiniBatchKMeans(n_clusters=3, random_state=0).fit(X)
    cost = (kmeans.inertia_)/ len(X)
    mu = kmeans.cluster_centers_
    m = len(X)
    c = kmeans.labels_
    print('J = ' + str(kmeans.inertia_ / m))





 
# lyrics-experiment.py
# Aumit Leon and Mariana Echeverria 

import numpy as np 
import csv


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

    return (words, content)


def create_vectors(list_dict, num_words):
    """
    Returns a list with numpy vectors of data 
    """
    x = [] # list that will hold data 

    for d in list_dict:
        # initializing numpy vector
        # it contains 5,000 (number of words) zeros
        temp = np.zeros(num_words)
        for key, val in d.items():
            key -= 1 # indexing in data starts at 1
            temp[key] = val # adding word and its frequency to vector 

        x.append(temp) # appends vector to x 
    
    return x


if __name__ == "__main__":
    (words, content) = read_txt('mxm_dataset_test.txt')
    print(len(content))
    #write_data('output2.csv')
    print(content[1])
    x = create_vectors(content, 5000)
    ex = x[1]
    for el in ex:
        print(el)
    #print(x[1])
    
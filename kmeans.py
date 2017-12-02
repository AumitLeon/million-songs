# 
# kmeans using sci-kit

from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np 
import csv
import random


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
            if key < num_words:
                key -= 1 # indexing in data starts at 1
                temp[key] = 1 # adding word and its frequency to vector 
            # temp[key] = val
        x.append(temp) # appends vector to x  

    return x


def run_various_Ks(x, K):
    """
    Runs algorithm for K = 1...30 and keeps
    track of the minimum costs J obtained. 
    """
    m = len(x) # length of data points
    min_list = [] # list that will contain minimum costs
    Ks = [i for i in range(1,K+1)] # values of K's

    for i in range(1, K+1):
        # runs algorithm with different values of K
        kmeans = KMeans(n_clusters=i, random_state=0).fit(x)
        minval = kmeans.inertia_
        print(minval)
        min_list.append(minval) # appends minimum cost 

    # Plotting J vs. K to choose best value of K
    plt.plot(Ks, min_list)
    plt.plot(Ks, min_list, '-o')
    plt.xlabel('K (# of clusters)')
    plt.ylabel('Cost function J')
    plt.title('J vs. K plot')
    plt.show()



#X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
# X = [np.array([0,500]),np.array([0, 459]), np.array([1, 2]), np.array([0,4])]
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# print(kmeans.labels_)
# # array([0, 0, 0, 1, 1, 1], dtype=int32)
# print(kmeans.predict(X))
# # array([0, 1], dtype=int32)
# print(kmeans.cluster_centers_)
# # array([[ 1.,  2.],[ 4.,  2.]])

if __name__ == "__main__":
    #(words, test_list, track_id, mxm_tid) = read_txt('mxm_dataset_test.txt')
    #print('== Finished test txt')
    (words, train_list, track_id, mxm_tid) = read_txt('mxm_dataset_train.txt')
    print('== Finished train txt')
    #print(words)
    #x_test= create_vectors(train_list, 5000)
    x_train = create_vectors(train_list, 5000)
    #X = np.array(x_train + x_test)
    #print(X)
    #print(type(X))

    #kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    #print(kmeans.labels_)
    #print(kmeans.cluster_centers_)
    #run_various_Ks(x_test, 30)
    print('== Beginning clustering')
    kmeans = MiniBatchKMeans(n_clusters=3, random_state=0).fit(x_train)
    minval = kmeans.inertia_
    print(minval)


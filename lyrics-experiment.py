# lyrics-experiment.py
# Aumit Leon and Mariana Echeverria 

import numpy as np 
import csv
import random
import matplotlib.pyplot as plt



# ===  ORGANIZING DATA 

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
        temp = np.zeros(num_words)
        for key, val in d.items():
            key -= 1 # indexing in data starts at 1
            temp[key] = val # adding word and its frequency to vector 

        x.append(temp) # appends vector to x  

    return x


# ===  HW 5 CODE (K-Means)

def squared_distance(vector_1, vector_2):
    """
    Computes the squared distance between two vectors.
    """
    # computing difference
    distance = vector_1 - vector_2
    # squaring difference 
    squared = np.dot(np.transpose(distance), distance)

    return squared


def closest_cluster(xi, mu):
    """
    Takes a data point xi and the list of centroids mu 
    and returns the index of the closest cluster center 
    for xi.
    """
    K = len(mu) # number of clusters
    # initializing distance to closest cluster 
    minval = squared_distance(xi, mu[0])
    # initializing index of cluster
    ci = 0
    
    # loops through clusters and fins closest custer 
    # center for xi
    for k in range(1, K):
        # computing squared distance between xi and clusterk
        dist = squared_distance(xi, mu[k])
        # compares dist to previous min distance 
        if dist < minval:
            minval = dist # new closest cluster center
            ci = k # new index 

    return ci


def cluster_centroids(x, c, K):
    """
    Takes the list of data points x, 
    the list of cluster indices c, and K.
    
    Returns a list mu containing the K centroids 
    based on the current assignment c.
    """
    # temporary list used to compute the mean of
    # data points assigned to a specific cluster 
    temp = []
    mu = [0]*K # initializing mu to empty list

    for k in range(K):
        for i in range(len(c)):
            # ci to which xi is assigned is equal to
            # current k
            if c[i] == k:
                # append data point to temp list
                temp.append(x[i])

        # add mean of data points assigned to cluster k
        mu[k] = sum(temp) / len(temp)
        temp = [] # clear temp list
    
    # returns mu as a numpy array
    return np.asarray(mu)


def J(x, c, mu):
    """
    Takes x, c, and mu as input.
    
    Returns the current cost J.
    """
    m = len(x) # number of data points
    res = 0 # initializing cost 

    for i in range(m):
        # index to which data point xi is assigned
        k = c[i] 
        # adding squared distance to result 
        res += squared_distance(x[i], mu[k])
    
    return res / m


def k_means_rand(x, K):
    """
    Takes x and K as input and runs the K-means algorithm. 
    
    This function initializes the cluster centers mu
    to K random samples from x.
    """
    mu = random.sample(x, K) # K random samples from x
    m = len(x) # length of data points
    c = [0]*m # initializing c 
    prev = [] # will keep track of previous c
    max_iter = 500

    for j in range(max_iter):
        for i in range(m):
            c[i] = closest_cluster(x[i], mu)
        #print("cost after cluster assignment: " + str(J(x, c, mu)))

        if c == prev:
            #print("converged after " + str(j) + " iterations")
            break
        else:
            mu = cluster_centroids(x, c, K)
            #print("cost after updating centroids: " + str(J(x, c, mu)))
            prev = c[:]
        
    return (mu, c)


def run(x, K):
    """
    Runs randomized K-means function for a given number 
    of repetitions and keeps track of the minimum cost 
    J obtained in each run. 
    """
    repeat = 100
    dict_count = dict()
    dict_values = dict()
    count = 0

    for i in range(repeat):
        # runs randomized K-means
        (mu, c) = k_means_rand(x, K)
        # computing cost
        cost = J(x, c, mu)

        # keeping track of how many times that minimum 
        # value was obtained
        if cost in dict_count:
            # increase count
            dict_count[cost] += 1
        else:
            # adds 1 to count dict and a tuple (mu, c) to 
            # dictionary containg mu, c values of each cost 
            dict_count[cost] = 1
            dict_values[cost] = (mu,c)
    
    min_cost = min(dict_count) # obtaining minimum cost
    num_min = dict_count[min_cost] # number of times minimum was found
    (min_mu, min_c) = dict_values[min_cost] # (mu,c) of minimum

    # ** used for debugging purposes 
    # determines size of USA's cluster
    # index_usa = 184
    # c_usa = min_c[index_usa] 

    # for el in min_c:
    #     if el == c_usa:
    #         count += 1

    print("k=" + str(K) + ", cost=" + str(min_cost) + " (" + str(num_min) + "/100)")

    return (min_cost, min_mu, min_c)


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
        (min_val, min_mu, min_c) = run(x,i)
        min_list.append(min_val) # appends minimum cost 

    # Plotting J vs. K to choose best value of K
    plt.plot(Ks, min_list)
    plt.plot(Ks, min_list, '-o')
    plt.xlabel('K (# of clusters)')
    plt.ylabel('Cost function J')
    plt.title('J vs. K plot')
    plt.show()


if __name__ == "__main__":
    (words, test_list, track_id, mxm_tid) = read_txt('mxm_dataset_test.txt')
    #(words, train_list) = read_txt('mxm_dataset_train.txt')
    
    x = create_vectors(test_list, 5000)
    
    run_various_Ks(x[:1000],30)
    
    
    # print(len(content))
    # #write_data('output2.csv')
    # print(content[1])
    # x = create_vectors(content, 5000)
    # ex = x[1]
    # for el in ex:
    #     print(el)
    # #print(x[1])
    
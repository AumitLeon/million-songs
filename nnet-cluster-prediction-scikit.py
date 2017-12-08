# nnet-cluster-prediction-scikit.py
# Predicts cluster to which each song was assigned to
# (based on their lyrics) using other features, such as:
# year,hotttness, tempo, loudness, etc.

import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
import numpy as np


# Scikit learn gives a depreaction warning for the rank function
# Not an error, but still annoting. These lines will suppress that output.
import warnings
warnings.filterwarnings("ignore")

def get_data():
    """
    Read features into vectors
    """
    # Initializing variables
    labels = []
    examples = []
    years_0 = []
    years_1 = []
    years_2 = []
    print "GETTING DATASET"
    print
    # Replace filename with the path to the CSV where you have the year predictions data saved.
    filename = "/Users/marianaecheverria/hotttness/million-songs/output-lyrics-year.csv"

    with open(filename, 'r') as f:
        for line in f:
            content = line.split(",")
            
            labels.append(content[0])
            label = int(content[0])
            year = int(content[1])
            # used to count years in each cluster
            if year != 0:
                if label == 0:
                    years_0.append(year)
                elif label == 1:
                    years_1.append(year)
                else:
                    years_2.append(year)
                    
            content.pop(0)

            # If we wanted pure lists
            content = [float(elem) for elem in content]
            #content = map(float, content)

            # If we want a list of numpy arrays, not necessary
            #npa = np.asarray(content, dtype=np.float64)

            examples.append(content)
    
    return (labels, examples, years_0, years_1, years_2)


def split_and_scale_data(labels, examples):
    """
    Splitting data in train and test sets and
    scaling each feature
    """
    print "SPLITTING TRAINING AND TEST SETS"
    print 
    # Turning lists into numpy arrays
    total_array = np.array(examples)

    # Scale the features so they have 0 mean
    total_scaled = preprocessing.scale(total_array)

    # Numpy array of the labels 
    total_labels = np.array(labels)
    # 
    # Split training and test:
    # Increase or decrease these sizes
    # Currently using first 10000 examples as training data
    # Last 1000 as test data
    training_examples = total_scaled[:2000]
    #training_examples = random.sample(total_array, 10)
    training_labels = total_labels[:2000]

    # Use the following 1000 examples as text examples
    test_examples = total_scaled[1500:]
    test_labels = total_labels[1500:]

    return (training_examples, training_labels, test_examples, test_labels)


def plot_histogram_years(years_0, years_1, years_2):
    """
    Plots histograms of years for each cluster
    """
    array_years0 = np.array(years_0)
    plt.figure(1)
    plt.hist(array_years0, bins = 100)
    plt.title('Years Cluster 0')
    plt.xlabel('years')
    plt.ylabel('frequency')
    plt.savefig('k_10cluster0.png')

    array_years1 = np.array(years_1)
    plt.figure(2)
    plt.hist(array_years1, bins = 100)
    plt.title('Years Cluster 1')
    plt.xlabel('years')
    plt.ylabel('frequency')
    plt.savefig('k_10cluster1.png')

    array_years2 = np.array(years_2)
    plt.figure(3)
    plt.hist(array_years2, bins = 100)
    plt.title('Years Cluster 2')
    plt.xlabel('years')
    plt.ylabel('frequency')
    plt.savefig('k_10cluster2.png')



if __name__ == "__main__":
    # getting data
    (labels, examples, years_0, years_1, years_2) = get_data()
    # splitting sets
    (training_examples, training_labels, test_examples, test_labels) = split_and_scale_data(labels, examples)

    # classifier
    clf = MLPClassifier(solver='sgd', alpha=0.00001,
                        hidden_layer_sizes=(50, 50, 50, 50, 50, 50, 50, 50, 20), random_state=1, \
                        batch_size = 100, learning_rate="adaptive", max_iter=200, \
                        momentum=0.9)
    # training model
    clf.fit(training_examples, training_labels)                         

    # getting predcitions
    y_pred = clf.predict(test_examples)

    # printing results
    accuracy = accuracy_score(test_labels, y_pred) * 100
    print "Accuracy of the model is: " + str(accuracy) + "%"
    print "Score: " + str(clf.score(test_examples, test_labels))
    print "Precision, recall and f-score:"
    print precision_recall_fscore_support(test_labels, y_pred, average="micro")
    # uncomment to plot histograms
    #plot_histogram_years(years_0, years_1, years_2)
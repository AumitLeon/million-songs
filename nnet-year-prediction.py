# Crude neural net for year prediction (7.5% accuracy)
# Script requires sci-kit learn. To install: 
# Data available at: http://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
# 90 features per example
# 12 = timbre averae
# 78 = timbre covariance
# Uses a subset of the full dataset to train faster
#
#
#
# HOW TO CONVERT TEXT FILE TO CSV (once dataset is downloaded)
# cd into the Directory where you have the dataset downloaded
# Run the following: 
# head YearPredictionsMSD.txt > yp_test.csv
# Docs for sci-kitlearn: http://scikit-learn.org/stable/modules/neural_networks_supervised.html

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
import numpy as np

# Scikit learn gives a depreaction warning for the rank function
# Not an error, but still annoting. These lines will suppress that output.
import warnings
warnings.filterwarnings("ignore")

## Read features into vectors 

labels = []
examples = []

# Replace filename with the path to the CSV where you have the year predictions data saved.
filename = "/mnt/c/Users/Aumit/Desktop/YearPredictionMSD.txt/yp.csv"
with open(filename, 'r') as f:
    for line in f:
        content = line.split(",")
        
        labels.append(content[0])

        content.pop(0)

        # If we wanted pure lists
        #content = [float(elem) for elem in content]
        #content = map(float, content)

        # If we want a list of numpy arrays, not necessary
        #npa = np.asarray(content, dtype=np.float64)

        examples.append(content)

# Turning lists into numpy arrays
total_array = np.array(examples)

# Scale the features so they have 0 mean
total_scaled = preprocessing.scale(total_array)

# Numpy array of the labels 
total_labels = np.array(labels)

# Split training and test:
# Increase or decrease these sizes
# Currently using first 10000 examples as training data
# Last 1000 as test data
training_examples = total_scaled[:10000]
#training_examples = random.sample(total_array, 10)
training_labels = total_labels[:10000]

# Use the following 1000 examples as text examples
test_examples = total_scaled[10000:11000]
test_labels = total_labels[10000:11000]


# Debugging
#X = [[0., 0.], [1., 1.]]
#y = [0, 1]

# Solver can be lbfgs, sgd, adam
# We can tune the hidden layers sizes
# 50, 50, 20 with 100 batch size worked well
# 0.0000001 9.4%
# 0.00001
clf = MLPClassifier(solver='sgd', alpha=0.00001,
                     hidden_layer_sizes=(50, 50, 20), random_state=1, batch_size = 100, learning_rate="adaptive", max_iter=200,
                     momentum=0.9)

clf.fit(training_examples, training_labels)                         


#Debugging
#print clf.predict([[2., 2.], [-1., -2.]])

# Acuracy given these lists should be 0.75
#y_pred = [0, 2, 1, 3]
#y_true = [0, 1, 1, 3]

y_pred = clf.predict(test_examples)
accuracy = accuracy_score(test_labels, y_pred) * 100
print "Accuracy of the model is: " + str(accuracy) + "%"
print "Score: " + str(clf.score(test_examples, test_labels))
print "Precision, recall and f-score:"
print precision_recall_fscore_support(test_labels, y_pred, average="micro")

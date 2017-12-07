


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
print "GETTING DATASET"
print
# Replace filename with the path to the CSV where you have the year predictions data saved.
filename = "/Users/marianaecheverria/Desktop/output-lyrics1.csv"
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
training_examples = total_scaled[:1500]
#training_examples = random.sample(total_array, 10)
training_labels = total_labels[:1500]

# Use the following 1000 examples as text examples
test_examples = total_scaled[1500:]
test_labels = total_labels[1500:]



clf = MLPClassifier(solver='sgd', alpha=0.00001,
                     hidden_layer_sizes=(50, 50, 50, 50, 50, 50, 50, 50, 20), random_state=1, batch_size = 100, learning_rate="adaptive", max_iter=200,
                     momentum=0.9)

clf.fit(training_examples, training_labels)                         


y_pred = clf.predict(test_examples)
print y_pred[:100]
print test_labels[:100]
accuracy = accuracy_score(test_labels, y_pred) * 100
print "Accuracy of the model is: " + str(accuracy) + "%"
print "Score: " + str(clf.score(test_examples, test_labels))
print "Precision, recall and f-score:"
print precision_recall_fscore_support(test_labels, y_pred, average="micro")
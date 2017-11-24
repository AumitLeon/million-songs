# Crude SVM implementation
# Most of this data extraction code is from the nnet year prediction experiement
# Same training/test split applied to SVM
# Interestingly... same accuracy. 

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
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
        
        labels.append(int(content[0]))

        content.pop(0)

        content = [float(elem) for elem in content]

        # If we want a list of numpy arrays, not necessary
        #npa = np.asarray(content, dtype=np.float64)

        examples.append(content)


# Split training and test:
# Increase or decrease these sizes
# Currently using first 10000 examples as training data
# Last 1000 as test data
training_examples = examples[:10000]
training_labels = labels[:10000]

test_examples = examples[-1000:]
test_labels = labels[-1000:]

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(training_examples, training_labels)  


y_pred = clf.predict(test_examples)
accuracy = accuracy_score(test_labels, y_pred) * 100
print "Accuracy of the model is: " + str(accuracy) + "%"
print "Precision, recall and f-score:"
print precision_recall_fscore_support(test_labels, y_pred, average="micro")
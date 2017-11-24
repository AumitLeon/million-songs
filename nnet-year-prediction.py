# Crude neural net for year prediction (5% accuracy)
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
# head -2000 YearPredictionsMSD.txt > yp_test.csv
# The above command will create a CSV file with the first 2000 examples

from sklearn.neural_network import MLPClassifier
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
filename = "/mnt/c/Users/Aumit/Desktop/YearPredictionMSD.txt/yp_test.csv"
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
training_examples = examples[:1500]
training_labels = labels[:1500]

test_examples = examples[-500:]
test_labels = labels[-500:]

# Debugging
#X = [[0., 0.], [1., 1.]]
#y = [0, 1]

# Solver can be lbfgs, sgd, adam
clf = MLPClassifier(solver='sgd', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(training_examples, training_labels)                         
MLPClassifier(activation='relu', alpha=0.001, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

#Debugging
#print clf.predict([[2., 2.], [-1., -2.]])

# Acuracy given these lists should be 0.75
#y_pred = [0, 2, 1, 3]
#y_true = [0, 1, 1, 3]

y_pred = clf.predict(test_examples)
accuracy = accuracy_score(test_labels, y_pred) * 100
print "Accuracy of the model is: " + str(accuracy) + "%"
print "Precision, recall and f-score:"
print precision_recall_fscore_support(test_labels, y_pred, average="micro")

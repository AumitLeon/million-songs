# Crude SVM implementation
# Most of this data extraction code is from the nnet year prediction experiement
# Takes a long time to run. CV goes up to 31% accuracy when tuning, but that doesen't carry over to the extra test set. 
# Need more experimentation.
# Same training/test split applied to SVM
# Interestingly... same accuracy. 

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
import optunity
import optunity.metrics
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

       # content = [float(elem) for elem in content]

        # If we want a list of numpy arrays, not necessary
        #npa = np.asarray(content, dtype=np.float64)

        examples.append(content)

# Convert list of examples to a numpy array
total_array = np.array(examples)

# Scale training examples
total_scaled = preprocessing.scale(total_array)

# Convert list of labels to numpy array
total_labels = np.array(labels)

# Split training and test:
# Increase or decrease these sizes
# Currently using first 10000 examples as training data
# Last 1000 as test data

# Training data
x_train_data = total_scaled[:5000]
y_train_data = total_labels[:5000]

#CV
x_train = total_scaled[5000:6000]
y_train = total_labels[5000:6000]

#test
x_test = total_scaled[6000:8000]
y_test = total_labels[6000:8000]

# Another test
test_examples = total_scaled[8000:10000]
test_labels = total_labels[8000:10000]



# Perform 10 fold cross validation to rune parameters
# score function: twice iterated 10-fold cross-validated accuracy
# original: num_folds = 10
@optunity.cross_validated(x=x_train_data, y=y_train_data, num_folds=3, num_iter=1)
def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = svm.SVC(C=10 ** logC, gamma=10 ** logGamma).fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    y_pred_vals = model.predict(x_test)
    #print decision_values
    acc_cv = accuracy_score(y_test, y_pred_vals)
    print acc_cv
    return acc_cv
    #return optunity.metrics.roc_auc(y_test, decision_values)

# perform tuning
hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-5, 1])

# train model on the full training set with tuned hyperparameters
optimal_model = svm.SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(x_train_data, y_train_data)




#clf = svm.SVC(decision_function_shape='ovo')
#clf.fit(training_examples, training_labels)  


y_pred = optimal_model.predict(test_examples)
accuracy = accuracy_score(test_labels, y_pred) * 100
print "Accuracy of the model is: " + str(accuracy) + "%"
print "Score: " + str(optimal_model.score(test_examples, test_labels))
print "Precision, recall and f-score:"
print precision_recall_fscore_support(test_labels, y_pred, average="micro")
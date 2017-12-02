import keras
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np
from sklearn import preprocessing
from keras import regularizers

from keras.utils import np_utils, generic_utils




model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=90))
model.add(Dense(units=100, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Dense(units=100, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Dense(units=100, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Dense(units=100, activation='relu'))
#model.add(Flatten())
model.add(Dense(units=2011, activation='softmax'))


#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
labels = []
examples = []
print "GETTING DATASET"
print
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
training_examples = total_scaled[:10000]
#training_examples = random.sample(total_array, 10)
training_labels = total_labels[:10000]

# Use the following 1000 examples as text examples
test_examples = total_scaled[10000:11000]
test_labels = total_labels[10000:11000]




#y_train, y_test = [np_utils.to_categorical(x) for x in (training_labels, test_labels)]

print "****Shape of y_train:" + str(training_examples.shape)

y_train = keras.utils.to_categorical(training_labels, num_classes=2011)

y_test = keras.utils.to_categorical(test_labels, num_classes=2011)
#y_train = training_labels.reshape(1000,1)
#y_test = test_labels.reshape(200,1)

model.fit(training_examples, y_train, epochs=200, batch_size=32)

loss_and_metrics = model.evaluate(test_examples, y_test, batch_size=32)
print loss_and_metrics
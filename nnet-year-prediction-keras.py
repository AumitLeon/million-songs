# Year prediction using Keras neural nets
# This is the baseline file we used to run our experiments on condor.
# Every experiment on condor uses most of this code, with a few small modifications catered to that particular experiment. 

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np
from sklearn import preprocessing
from keras import regularizers
from keras.utils import np_utils, generic_utils


# Defining the sequential model
model = Sequential()

# Our examples of 90 features, so input_dim = 90
model.add(Dense(units=100, activation='relu', input_dim=90))
model.add(Dense(units=100, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Dense(units=100, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Dense(units=100, activation='relu', kernel_regularizer=regularizers.l2(0.00001)))
model.add(Dense(units=100, activation='relu'))
#model.add(Flatten())

# Output is 0-2011, after conversion to categorical vars
model.add(Dense(units=2011, activation='softmax'))

# Tune the optimizer
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

        # If we wanted pure lists, and convert from string to float
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

# Split training and test:
# Increase or decrease these sizes affects run-time
training_examples = total_scaled[:10000]
#training_examples = random.sample(total_array, 10)
training_labels = total_labels[:10000]

# Use the following 1000 examples as text examples
test_examples = total_scaled[10000:11000]
test_labels = total_labels[10000:11000]

# Convert to categorical in order to produce proper output
y_train = keras.utils.to_categorical(training_labels, num_classes=2011)

y_test = keras.utils.to_categorical(test_labels, num_classes=2011)

# Train the model!
model.fit(training_examples, y_train, epochs=200, batch_size=32)

# Loss and metrics
loss_and_metrics = model.evaluate(test_examples, y_test, batch_size=32)
print loss_and_metrics

print ("Creating Plots!")
print (history_1.history.keys())

#accuracy
plt.figure(1)
plt.plot(history_1.history['acc'])
plt.plot(history_1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig("model_acc.png")

#loss
plt.figure(2)
plt.plot(history_1.history['loss'])
plt.plot(history_1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.savefig("model_loss.png")
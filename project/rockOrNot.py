"""
Function h(x) guesstimates whether a given song, or set of features x, is genre Y (or not).

1 'Pop_Rock'
2 'Electronic'
3 'Rap'
4 'Jazz'
5 'Latin'
6 'RnB'
7 'International'
8 'Country'
9 'Reggae'
10 'Blues'

There are several commented out blocks here,
- one produces bar charts with predicted labels
- one produces and plots a confusion matrix

You can adjust the amount of layers and the amount of nodes in them
by adjusting the variable N, which you can find in the begigning of main
(to find main just go Ctrl + F "# Main #" )

"""
import itertools
import os
import csv
import matplotlib.pyplot as plt
import plot_barchart
import numpy as np
# the neural network itself
from sklearn.neural_network import MLPClassifier as mpl
# Neural networks are sensitive to feature scaling, so we'll preprocess the data a bit more I guess
from sklearn.preprocessing import StandardScaler as sclr
# Reporting tools included in sklearn
from sklearn.metrics import classification_report as cr,confusion_matrix as cm, log_loss as ll
from sklearn.model_selection import train_test_split

# Firstly let's define some vars
# Starting with paths, just for the sake of convenience and readability
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

data_path = os.path.join(__location__, "data/")
train_data_path = data_path + "train_data.csv"
train_label_path = data_path + "train_labels.csv"
test_data_path = data_path + "test_data.csv"
dummy_path = data_path + "dummy"

# Then the labels, this is from the project description paper
labels = ['Pop_Rock',
          'Electronic',
          'Rap',
          'Jazz',
          'Latin',
          'RnB',
          'International',
          'Country',
          'Reggae',
          'Blues']

# Create Neural Network
## Allow dumping of weights to file?
### (this way only train once)

# Reading the csv
## Write line number and prediction in solution.csv

#First, let's analyze the data a bit
# We'll draw a bar diagram with the sums of all the labels

# Function for reading the CSV files
def readCSV(data_path):
    countingvarlimit = 100000
    data_content = []
    with open(data_path, 'r') as csvfile:
        countingvar = 0
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data_content.append([float(i) for i in row])
            if countingvar > countingvarlimit:
                break
            countingvar += 1
#    return data_content, data_labels
    return data_content


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

################################## Main ##################################

# This is how many layers (and perceptrons per layer) we'll have
N = (100)

# Read data into memory
training_data = readCSV(train_data_path)
training_labels = readCSV(train_label_path)
test_data = readCSV(test_data_path)

# Scale the data
scaler = sclr()
# Fit only to training
scaler.fit(training_data)

# Create the classifier, first one will have 10 layers
# Second 10, 10
# Third 10, 10, 10
# Fourth 100
# Fifth 100, 100
# Sixth 100, 100, 100
# Seventh 1000
# Eight 1000, 1000
# Ninth 1000, 1000, 1000
classifier = mpl(N, max_iter=500)
# Set these, just needed to be this I guess
classifier.out_activation_ = "Softmax"
classifier.n_outputs_ = 10

classifier.fit(training_data, training_labels)

predictions = classifier.predict(test_data)

#Sample_id,Sample_label
linecounter = 1
# Write predictions to a .csv
writer = csv.writer(open("solution_accuracy_"+str(N)+".csv", "wb"))
for i in list(predictions):
    writer.writerow([linecounter,int(i)])
    linecounter += 1
'''
# 1st line: Sample_id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9,Class_10
# Then do the logloss
writer2 = csv.writer(open("solution_logloss.csv", "wb"))
proba = classifier.predict_proba(test_data)
linecounter = 1
for row in proba:
    #print str(list(row)).strip("[]")
    line = []
    line.append(linecounter)
    for e in list(row):
        line.append(float(e))
    writer2.writerow(line)
    linecounter += 1
    #if linecounter > 10:
    #    break
'''    

#np.savetxt('solution.csv', ([ int(x) for x in predictions ]), delimiter=',')

#Enable bar chart here!
'''

# Plot bar chart

label_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
# Plot the labels of the training data, this part can be removed later
for i in list(predictions):
    label_count[int(i)] = label_count[int(i)] + 1
#print label_count
#print list(label_count.values())
plot_barchart.plot_bar(list(label_count.values()), "N = "+str(N)+", test")

# Produce confusion matrix
# Split the training set, that's how we'll get both y_true and y_pred
#x_true = training_data[0:len(training_data)/2]
#y_true = training_labels[0:len(training_data)/2]
#x_the_rest = training_data[len(training_data)/2:len(training_data)]
#y_the_rest = training_labels[len(training_labels)/2:len(training_labels)]
''' #Enable bar chart


'''
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(training_data, training_labels, random_state=0)

#print y_the_rest

#scaler.fit(x_the_rest)
classifier2 = mpl(N, max_iter=500)
classifier2.fit(X_train, y_train)
y_pred = classifier2.predict(X_test)

# Compute confusion matrix
cnf_matrix = cm(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names,
#                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
'''
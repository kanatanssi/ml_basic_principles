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
"""
import os
import csv
import matplotlib.pyplot as plt
import plot_barchart
# the neural network itself
from sklearn.neural_network import MLPClassifier as mpl
# Reporting tools included in sklearn
from sklearn.metrics import classification_report,confusion_matrix

# Firstly let's define some vars
# Starting with paths, just for the sake of convenience and readability
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

data_path = os.path.join(__location__, "data/")
train_data_path = data_path + "train_data.csv"
train_label_path = data_path + "train_labels.csv"
test_data_path = data_path + "test_data.csv"

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
    #countingvarlimit = 10000
    data_content = []
    with open(data_path, 'r') as csvfile:
    #    countingvar = 0
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data_content.append([int(i) for i in row])
    #        if countingvar > countingvarlimit:
    #            break
    #        countingvar += 1
#    return data_content, data_labels
    return data_content

# Main
training_data = readCSV(train_data_path)
training_labels = readCSV(train_label_path)
test_data = readCSV(test_data_path)

#label_count = {}
# Plot the labels of the training data, this part can be removed later
#for label in readCSV(train_label_path):
#    label_count[label[0]] = label_count.get(label[0], 0) + 1
#plot_barchart.plot_bar(list(label_count.values()),"Training")
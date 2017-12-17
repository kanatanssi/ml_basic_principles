'''task 1
'''
import glob
import os
import numpy as np
import matplotlib.pyplot as mpl
from PIL import Image
from sklearn.cluster import KMeans
import sklearn

############################# GET REDNESS AND GREENNESS OF IMAGES #############################
"""
summers = np.array([[192625102, 208475564],
                [253880037, 277516044],
                [205009227, 214408574],
                [274764712, 264447209],
                [229547016, 238525595],
                [214962580, 230902788],
                [238842115, 253202518],
                [236707748, 250308138],
                [210443387, 246156556],
                [224688414, 243918203]])

winters = np.array([[182680194, 206532057],
                [130669935, 126203295],
                [193437910, 205465094],
                [235671920, 246064091],
                [215218169, 232822175],
                [194143151, 230591553],
                [220259929, 233483127],
                [240708874, 277754344],
                [194761890, 257931445],
                [143712732, 146566283]])
"""
summers = np.array([[ 100.32557396,  108.58102292],
                    [ 132.22918594,  144.53960625],
                    [ 106.77563906,  111.67113229],
                    [ 143.10662083,  137.73292135],
                    [ 119.5557375,   124.23208073],
                    [ 111.95967708,  120.26186875],
                    [ 124.3969349,   131.87631146],
                    [ 123.28528542,  130.36882188],
                    [ 109.60593073,  128.20653958],
                    [ 117.02521563,  127.04073073]])

winters = np.array([[ 95.14593438,  107.56877969],
                    [ 131.28329053,  126.79568444],
                    [ 100.74891146,  107.01306979],
                    [ 122.74579167,  128.15838073],
                    [ 112.09279635,  121.26154948],
                    [ 101.11622448,  120.09976719],
                    [ 114.71871302,  121.60579531],
                    [ 125.36920521,  144.66372083],
                    [ 101.43848438,  134.33929427],
                    [ 116.9537207 ,  119.27594645]])
#M = 10

seasons = ["summer1", "summer10", "summer2", "summer3", "summer4", "summer5", "summer6", "summer7", "summer8", "summer9", "winter1", "winter10", "winter2", "winter3", "winter4", "winter5", "winter6", "winter7", "winter8", "winter9"]
purities = []

# Do the whole thing 10 times, with different values for M between 1 and 11
for M in range(300,310):

    pred = KMeans(n_clusters=2, n_init=1, max_iter=M, random_state=None).fit_predict(np.vstack((summers, winters)))

    #print pred

    # these are used for purity
    summer_sum = 0.0
    winter_sum = 0.0
    summer_correct = 0.0
    winter_correct = 0.0
    winter = 0

    # Summer1 seems to be reliably classified as summer, so I'm benchmarking on that :d
    summer = pred[1]
    if summer == 0:
        winter = 1
    
    # Truths
    y_true = [summer, summer, summer, summer, summer, summer, summer, summer, summer, summer, winter, winter, winter, winter, winter, winter, winter, winter, winter, winter]

    for i in range(len(pred)):
        if pred[i] == summer:
            summer_sum += 1
            print "Sample " + str(i) + " Classified as summer, actual: " + str(seasons[i])
            if pred[i] == y_true[i]:
                summer_correct += 1
        else:
            winter_sum += 1
            print "Sample " + str(i) + " Classified as winter, actual: " + str(seasons[i])
            if pred[i] == y_true[i]:
                winter_correct += 1

    print "Total summers: " + str(summer_sum) + " total winters: " + str(winter_sum)
    print "Correct summers: " + str(summer_correct) + " correct winters: " + str(winter_correct)
    print "Accuracy summers: " + str(summer_correct/len(pred)) + " accuracy winters: " + str(winter_sum/len(pred))

    # Print confusion matrix
    conf_mat = sklearn.metrics.confusion_matrix(y_true, pred)
    print "Confusion matrix:"
    print conf_mat

    # Get purity by dividing highest count of each cluster and dividing that with total number of samples
    # Purity does not show correctness of clustering
    #highest_sum = 0.0
    #for row in conf_mat:
    #    highest_sum += max(row)

    purity = (summer_correct + winter_correct)/len(pred)
    print "Purity: ", purity
    purities.append(purity)

# Plotting in case we want
#mpl.scatter(np.vstack((summers, winters))[:, 0], np.vstack((summers, winters))[:, 1], c=pred)
mpl.plot(range(10),purities)
mpl.xlabel("M")
mpl.ylabel("Purity")
mpl.show()

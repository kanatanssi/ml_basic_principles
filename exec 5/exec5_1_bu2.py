'''task 1
'''
import glob
import os
import numpy as np
import matplotlib.pyplot as mpl
from PIL import Image
from sklearn.cluster import KMeans

############################# GET REDNESS AND GREENNESS OF IMAGES #############################

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

img_path = os.path.join(__location__, 'images/')

filenames = []
imgs = np.array([[0, 0]])

# open each image in turn
for filename in glob.glob(img_path + '*.jpeg'):
    kuv = Image.open(filename)

	# get width and height of image
    width, height = kuv.size

    # For normalization by amount of pixels in each image
    pixels = width * height

	# Set sums to 0 at the begigning of an iteration
    sum_red = 0
    sum_green = 0

	# loop through each pixel in image
    for w_px in range(0, width):
        for h_px in range(0, height):
			# get colors of each pixel
            r, g, b, = kuv.getpixel((w_px, h_px))
            sum_red += r
            sum_green += g

    # Save filename without path to it (in case we want to use them later)
    filenames.append(filename.replace(img_path, '').replace('.jpeg', ''))
    
    # Normalize by amount of pixels before appending 
    sum_green = 1.0 / pixels * sum_green
    sum_red = 1.0 / pixels * sum_red
    
    # temporary var
    ddong = np.array([sum_red, sum_green])
    # Add redness and greenness to numpy array
    imgs = np.vstack((imgs, np.array([sum_red, sum_green])))

    #print filename.replace(img_path, '').replace('.jpeg', ''), 'Greenness: ', sum_green, 'Redness: ', sum_red

############################# DO THE KMEANS THING #############################

imgs = imgs[1:len(imgs)]
print imgs
"""
Redness and greenness already extracted:
imgs = np.array([192625102, 208475564],
                [253880037, 277516044],
                [205009227, 214408574],
                [274764712, 264447209],
                [229547016, 238525595],
                [214962580, 230902788],
                [238842115, 253202518],
                [236707748, 250308138],
                [210443387, 246156556],
                [224688414, 243918203],
                [182680194, 206532057],
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

M = 10
#kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
kmeans = KMeans(n_clusters=2, n_init=1, max_iter=M, random_state=None).fit_predict(np.vstack((summers, winters)))

mpl.subplot()
mpl.scatter(np.vstack((summers, winters))[:, 0], np.vstack((summers, winters))[:, 1], c=kmeans)

# Get centroids
#centroids = kmeans.cluster_centers_
#labels = kmeans.labels_
#mpl.scatter(centroids[:, 0],centroids[:, 1], marker = 'x', s=150, linewidths = 5, zorder = 10)

# Draw summers / winters
#for i in range(len(summers)):
#    mpl.scatter(summers[i][0], summers[i][1], marker='.', c = 'r')
#    mpl.scatter(winters[i][0], winters[i][1], marker='^', c = 'b')

mpl.show()
"""

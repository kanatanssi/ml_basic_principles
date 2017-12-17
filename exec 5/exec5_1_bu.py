'''task 1
'''
import glob
import os
import numpy as np
import matplotlib.pyplot as mpl
from PIL import Image
from sklearn.cluster import KMeans

############################# GET REDNESS AND GREENNESS OF IMAGES #############################
'''
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

img_path = os.path.join(__location__, 'images/')
#print PATH

# For storing sums of the colors
#GR = []
#RE = []
#imgs = []
imgs = np.array([[0, 0]])
filenames = []

# open each image in turn
#for i in range(1, 8):
    #kuv = Image.open(PATH + 'shot' + str(i) + '.jpeg')
#for kuv in Image.open(PATH + '*'):
print img_path
for filename in glob.glob(img_path + '*.jpeg'):
#    if filename == '/Users/ville/lipasto/ml_basic_principles/exec 5/images/winter4.jpeg':
#        break
#if(True): #for testing
#    filename = img_path +'winter9.jpeg'
#    print filename
    kuv = Image.open(filename)
	# get width and height of image
    width, height = kuv.size
	#print(width, height) #For debugging

	# Set sums to 0 at the begigning of an iteration
    sum_red = 0
    sum_green = 0

	# loop through each pixel in image
    for w_px in range(0, width):
        for h_px in range(0, height):
			# get colors of each pixel
            r, g, b, = kuv.getpixel((w_px, h_px))
#			print(r, g)
            sum_red += r
            sum_green += g

#    RE.append(sum_red)
#    GR.append(sum_green)
    # Save filename without path to it
    filenames.append(filename.replace(img_path, '').replace('.jpeg', ''))

    ddong = np.array([sum_red, sum_green])

    #imgs.append([sum_red, sum_green])
    imgs = np.vstack((imgs, np.array([sum_red, sum_green])))
    #imgs.append([sum_red, sum_green])
#    print 'pixel color sums:'
#    print 'red: and green:'
#    print sum_red, sum_green
    print imgs
    print filename.replace(img_path, '').replace('.jpeg', ''), 'Greenness: ', sum_green, 'Redness: ', sum_red

#if len(RE) == len(GR):
#    DIM = range(1, 8)
#mpl.plot(RE, GR, 'ro')
#mpl.xlabel('Redness')
#mpl.ylabel('Greenness')
#mpl.xlim([0.5, 7.5])
#mpl.show()

############################# DO THE KMEANS THING #############################

print 'Doing the kmeans thing'
imgs = imgs[1:len(imgs)]
print imgs
'''
'''
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
'''

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

#kmeans = KMeans(n_clusters=2)
#kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
#kmeans.fit_predict(np.vstack((summers,winters)))

M = 10

y_pred = KMeans(n_clusters=2, n_init=1, max_iter=M, random_state=None).fit_predict(np.vstack((summers, winters)))

#mpl.subplot(221)
mpl.subplot()
mpl.scatter(np.vstack((summers, winters))[:, 0], np.vstack((summers, winters))[:, 1], c=y_pred)

#centroids = kmeans.cluster_centers_
#labels = kmeans.labels_

#for i in range(len(summers)):
#    mpl.scatter(summers[i][0], summers[i][1], marker='.', c = 'r')
#    mpl.scatter(winters[i][0], winters[i][1], marker='^', c = 'b')

#for img in imgs:
#    mpl.scatter(img[0], img[1])
#mpl.scatter(imgs[0][1], imgs[0][2])
#mpl.scatter(imgs[1][1], imgs[1][2])
#mpl.scatter(centroids[:, 0],centroids[:, 1], marker = 'x', s=150, linewidths = 5, zorder = 10)
mpl.show()

"""Exercise 2.1

"""
import matplotlib.pyplot as mpl
import numpy as np
import glob
from PIL import Image

FILEPATH = "Webcam/"
FILENAME = "MontBlanc"

def read_images(path, name):
    """
    A function to read a set of images
    Reads all filenames from specified path
    Adds images with filenames containing the specified name to list
    Returns list of images
    """
    # For saving images in a list
    imgs = []
    
    # Get all files in a folder
    for filename in glob.glob(path + "*" + name + "*"):
        imgs.append(Image.open(filename))
        print 1
    return imgs

# Minutes from 07:00 for the images (visible on each image)
MINUTES = [211, 271, 121, 31, 341, 401, 241, 181, 301, 301]

# For storing greenness
GR = []

for i in range(1, 11):
    img = Image.open("Webcam/MontBlanc" + str(i) + ".png")
# Get the greenness of each image in the list returned by read_images 
#for img in read_images(FILENAME, FILEPATH):
    # Set sum_green to 0 at the beginging of each iteration
    sum_green = 0

    # get width and height of image
    width, height = img.size

    # For normalization by amount of pixels in each image
    pixels = width * height
    #print "Pixels " + str(pixels)

	# loop through each pixel in image
    for w_px in range(0, width):
        for h_px in range(0, height):
			# get colors of each pixel
            r, g, b, a = img.getpixel((w_px, h_px))
            sum_green += g
            #sum_green += (g - (0.5 * (r + b)))
            # Normalize by abount of pixels before appending
            #GR.append(1 / pixels * sum_green)
    #print "Sumgreen " + str(sum_green)
    # Normalize by abount of pixels before appending
    GR.append(1.0 / pixels * sum_green)

# Debugging. Print GR
#printingvar = 1
#for g in GR:
#    print "Greenness for MontBlanc" + str(printingvar)+ ".png: " + str(g)
#    printingvar = printingvar + 1

X = np.matrix([1 for i in range(0, 10)])
X = np.vstack([X, GR])
Y = np.matrix(MINUTES)
#print (X, '\n', Y)
W = (X * X.transpose()).I * X * Y.transpose()
#print W

# Linear reg func
#fit = np.polyfit(GR, MINUTES, 1)
#fn_fit = np.poly1d(fit)

#mpl.plot(GR, MINUTES, 'ro', GR, fn_fit(GR), 'k')
#mpl.xlabel('Greenness')
#mpl.ylabel('Minutes')
#mpl.show()

"""ASS2
"""
# a function for counting mean square error
# N - number of items to compare (images)
# y - labels
# h - hypotheses
def mean_sq_error(N, y, h):
    er = 0
    for i in range(0, N):
        er += (y[i] - h[i])*(y[i] - h[i])
    er /= N
    return er


hx = []
wa = W.item(0)
wb = W.item(1)
# count time with W
for g in GR:
    h = wb * g + wa
    hx.append(h)

# count error:
error = mean_sq_error(10, MINUTES, hx)
print "Mean squared error: " + str(error)

#def rmse(predictions, targets):
#    return np.sqrt(((predictions - targets) ** 2).mean())

#error = rmse(hx, Y)
#print "Root mean squared error: " + str(error)

"""ASS3
"""

# new weights
X = np.matrix([1 for i in range(0, 10)])
X = np.vstack([X, GR])
Y = np.matrix(MINUTES)
#print (X, '\n',  Y)
W32 = (X * X.transpose() + 10 * 2 * np.identity(2)).I * X * Y.transpose()
W35 = (X * X.transpose() + 10 * 5 * np.identity(2)).I * X * Y.transpose()

"""ASS4
"""

# OBS: 4 for problem number, 2 and 5 for lambda
hx42 = []
hx45 = []
wa42 = W32.item(0)
wb42 = W32.item(1)
wa45 = W35.item(0)
wb45 = W35.item(1)


# count times with W32 and W35
for x in GR:
    h42 = wb42 * x + wa42
    h45 = wb45 * x + wa45
    hx42.append(h42)
    hx45.append(h45)

#line_down, = plt.plot([3,2,1], label='Line 1')
#plt.legend(handles=[line_up, line_down])
data_points, = mpl.plot(GR, MINUTES, 'ro', label="Data points")
linear_reg, = mpl.plot(GR, hx, '-r', label="Linear regression")
lam2, = mpl.plot(GR, hx42, '-m', label="Lamda2")
lam5, = mpl.plot(GR, hx45, '-g', label="Lamda5")
#data_points, = mpl.plot(GR, MINUTES, 'ro', label="Data points")
#mpl.plot(GR, MINUTES, 'ro', GR, hx, '-r', GR, hx42, '-m', GR, hx45, '-g')
mpl.ylabel("Minutes")
mpl.xlabel("Greenness")
#mpl.legend([MINUTES, hx, hx42, hx45],["Data points","Linear regression","Lamda 2","Lamda 5"])
#mpl.legend(handles=["Data points","Linear regression","Lamda 2","Lamda 5"])
mpl.legend(handles=[data_points, linear_reg, lam2, lam5])
mpl.show()

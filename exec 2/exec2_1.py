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
#MINUTES = [211, 271, 121, 31, 341, 401, 241, 181, 301, 301]
MINUTES = [211, 271, 121, 31, 341, 401, 241, 181, 301, 301]
# Images in a list
#IMGS = read_images(FILENAME, FILEPATH)

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

#X = np.matrix([1 for i in range(0, 10)])
#X = np.vstack([X, GR])
#Y = np.matrix(MINUTES)
#print (X, '\n', Y)
#W = (X * X.transpose()).I * X * Y.transpose()
#print W

fit = np.polyfit(GR, MINUTES, 1)
fn_fit = np.poly1d(fit)

mpl.plot(GR, MINUTES, 'ro', GR, fn_fit(GR), 'k')
mpl.xlabel('Greenness')
mpl.ylabel('Minutes')
mpl.show()

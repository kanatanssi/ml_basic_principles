"""task 1
"""
#import glob
#mpl.use('TkAgg')
#import matplotlib.pyplot as mpl
import os
from PIL import Image

#TARGET_FOLDER = "~Desktop/code/\"Course_repo\"/Webcam/shot"

PATH = os.path.join(os.path.expanduser("~"), "Desktop", "code", "course_repo", "Webcam")
print PATH

# For summing REdness and GReenness in a pictuRE
#sum_red = 0
#sum_green = 0

# For storing sums of the colors
GR = []
RE = []

# open each image in turn
for i in range(1, 7):
    kuv = Image.open(PATH + "shot" + str(i) + ".jpg")

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

    RE.append(sum_red)
    GR.append(sum_green)
    print "pixel color sums:"
    print "red: and green:"
    print sum_red, sum_green

#	matplotlib.plot(range(1,8), sum_green, range(1,8), sum_red)

#mpl.ylabel('Color intensity')
#mpl.xlabel('Image')
#mpl.show()
for i in RE:
    print i

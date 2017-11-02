"""task 1
"""
import matplotlib.pyplot as mpl
from PIL import Image

PATH = "Webcam/"
#print PATH

# For storing sums of the colors
GR = []
RE = []

# open each image in turn
for i in range(1, 8):
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

#if len(RE) == len(GR):
#    DIM = range(1, 8)
mpl.plot(RE, GR, 'ro')
mpl.xlabel('Redness')
mpl.ylabel('Greenness')
#mpl.xlim([0.5, 7.5])
mpl.show()

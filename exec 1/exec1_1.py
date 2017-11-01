from PIL import Image as img
#from __future__ import print_function
import glob
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

target_folder = "webcam/*.jpg"
sum_red = 0
sum_green = 0
#sums = []

# gather all images in target folder into a list
kuv_list = []

for filename in glob.glob(target_folder):

	kuv = img.open(filename)

	# get width and height of image
	width, height = kuv.size

	#print(width, height)
	
	sum_red = 0
	sum_green = 0
	sums_both = []

	# loop through each pixel in image
#	for w_px in width:
	for w_px in range(0, width):
#		for h_px in height:
		for h_px in range(0, height):
			# get colors of each pixel
			r, g, b, = kuv.getpixel((w_px, h_px))
#			print(r, g)
			sum_red += r
			sum_green += g
	
	print("pixel color sums:")
	print("red: and green:")
	print([sum_red, sum_green])
	sums_both.append([sum_red, sum_green])


#	matplotlib.plot(range(1,8), sum_green, range(1,8), sum_red)

mpl.ylabel('Color intensity')
mpl.xlabel('Image')
mpl.show()
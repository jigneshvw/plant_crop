# import the necessary packages
import cv2
import numpy as np
import os
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path of the image")
args = vars(ap.parse_args())


# create mouse global coordinates
x_mouse = 0
y_mouse = 0                
 
# create thermal video fps variable (8 fps in this case)
# fps = 8
 
# mouse events function
cv2.namedWindow('gray8')

def mouse_events(event, x, y, flags, param):

	global x_mouse
	global y_mouse

    # mouse movement event
	if event == cv2.EVENT_MOUSEMOVE:
    # update global mouse coordinates

		x_mouse = x
		y_mouse = y

	print(x_mouse)
	print(y_mouse)

	cv2.circle(gray8_frame, (x_mouse, y_mouse), 2, (255, 255, 255), -1)
	# write temperature
	cv2.putText(gray8_frame, "{0:.1f} Fahrenheit, (Celsius: {1:.1f}) ".format(temperature_pointer, temperature_pointer_cel), (x_mouse - 40, y_mouse - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
	# show the thermal frame
	print("{0:.1f} Fahrenheit, (Celsius: {1:.1f})".format(temperature_pointer, temperature_pointer_cel))
	cv2.imshow("gray8", gray8_frame)

# set up mouse events and prepare the thermal frame display
# gray16_frame = cv2.imread(os.listdir(args["image"]), cv2.IMREAD_ANYDEPTH) 	
# gray16_frame = cv2.imread(args["image"], cv2.IMREAD_ANYDEPTH)
# cv2.imshow('gray8', gray16_frame)







if __name__=="__main__":

	# loop over the thermal video frames
	# image = os.listdir(args["image"])

	image = args["image"]

	    # filter .tiff files (gray16 images)
	# if image.endswith(".tiff"):
	    # define the gray16 frame path
	# file_path = os.path.join(args["image"], image)
	file_path = args["image"]

	# open the gray16 frame
	gray16_frame = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)	
	# cv2.imshow("gray11", gray16_frame)



	for counter in range(10000):
		# calculate temperature
		print(f'x coordinate: {x_mouse} and y coordinate: {y_mouse}')
		temperature_pointer = gray16_frame[y_mouse, x_mouse]
		temperature_pointer = (temperature_pointer / 100) - 273.15
		temperature_pointer = (temperature_pointer / 100) * 9 / 5 - 459.67
		temperature_pointer_cel = (temperature_pointer - 32) * 5/9


		# convert the gray16 frame into a gray8
		gray8_frame = np.zeros((120, 160), dtype=np.uint8)
		gray8_frame = cv2.normalize(gray16_frame, gray8_frame, 0, 255, cv2.NORM_MINMAX)
		gray8_frame = np.uint8(gray8_frame)
		# cv2.imshow("gray9", gray8_frame)


		# colorized the gray8 frame using OpenCV colormaps
		gray8_frame = cv2.applyColorMap(gray8_frame, cv2.COLORMAP_INFERNO)

		# write pointer



		# wait 125 ms: RGMVision ThermalCAM1 frames per second = 8
		# cv2.waitKey(int((1 / fps) * 1000))

		cv2.setMouseCallback('gray8', mouse_events)

		cv2.waitKey(1)

		# cv2.destroyAllWindows()



# Import Libraries 
from plantcv import plantcv as pcv
import matplotlib
import cv2
from detecto.core import Model #import detecto library to load trained model
from detecto import core, utils, visualize #import detecto library and its modules to load trained model
import pickle #import pickle to load classes
import imutils
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from flask import Flask, render_template, Response
# from flask import redirect, url_for

import os
import pymysql 


# * ---------- Create App --------- *
app = Flask(__name__)
CORS(app, support_credentials=True)

# @app.route('/')
# def index():
#   return render_template('index.html')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


FILE_PATH = os.path.dirname(os.path.realpath(__file__))
print(FILE_PATH)


@app.route('/growth_pred',methods=['GET','POST'])
# def growth_stage(detector_label_pkl, detector_model):
def growth_stage():
#         img_path = r"D:\plant_proj\growth_prediction\new_data\xml_data_valid\images\5.jpg"
#         img_path = "D:\plant_proj\growth_prediction\data\plant-1_task-6565_sv30_cabin-1.jpg"
#         img_path = "D:\plant_proj\growth_prediction\data\plant-1_task-6757_sv150_cabin-1.jpg"

	detector_label_pkl = request.args.get('detector_label_pkl', type=str ,default='') #get paramter value from url
	detector_model = request.args.get('detector_model', type=str ,default='') #get paramter value from url

   # To connect MySQL database 
	conn = pymysql.connect( 
		host='localhost', 
		user='username', 
		db='DB name', 
		password='password',
		port='port number'
		) 
	  
	cur = conn.cursor() 
	cur.execute("select * from growth_plant where processed_new = 'new'") 
	output_new_data = cur.fetchall() 
	# folder_name = output[0][1]
	# train_data_path = output[0][2]

	print('len output_new_data::: ', output_new_data)

	cur.execute("select * from growth_plant") 
	output_all_data = cur.fetchall() 

	print('len output_all_data::: ', output_all_data)

	# To close the connection 
	conn.close() 

	source_path = output_new_data[0][2]


	try:
		#create dataframe
		all_subfolders = []
		# image_path = '{path}/Data/Crop1-tomato/test/{image_folder}/{test}'.format(path=FILE_PATH, test=input_id, image_folder=folder_name_image) #testimages[i]----- (query image) change path as per server      
		# image_path = '{path}\\thermal_test\\'.format(path=FILE_PATH) #testimages[i]----- (query image) change path as per server
		image_path = '{path}'.format(path=source_path) #testimages[i]----- (query image) change path as per server

		for i in os.listdir(image_path):
			path = image_path + i
			all_subfolders.append(path)

		sub_fol_image_path_dict = {}
		for subdir in all_subfolders:
			all_image_path = []
			for (dirpath, dirnames, filenames) in os.walk(subdir): #change path as per the server
				for img_name in filenames:
					image_names = dirpath + '\\' + img_name
					all_image_path.append(image_names)
			sub_fol_image_path_dict[subdir.split('\\')[-1]] = all_image_path

		all_preds_len_height = {}  
		all_preds_leaf_cnt_pvc = {}  
		all_preds_leaf_cnt_contour = {}

		for ind, dir_name in enumerate(sub_fol_image_path_dict.keys()):
			print('dir_name', dir_name) 

			pred_values_len_height = {}
			pred_values_leaf_cnt_pvc = {}
			pred_values_leaf_cnt_countour = {}

			for ind, single_path in enumerate(sub_fol_image_path_dict[dir_name]):

				img_path = single_path
				# Read image
				# Inputs:
				#   filename - Image file to be read in 
				#   mode - How to read in the image; either 'native' (default), 'rgb', 'gray', or 'csv'
				img, path, filename = pcv.readimage(filename=img_path)


				# Convert RGB to HSV and extract the saturation channel
				# Inputs:
				#   rgb_image - RGB image data 
				#   channel - Split by 'h' (hue), 's' (saturation), or 'v' (value) channel
				s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')


				# Take a binary threshold to separate plant from background. 
				# Threshold can be on either light or dark objects in the image. 
				# Inputs:
				#   gray_img - Grayscale image data 
				#   threshold- Threshold value (between 0-255)
				#   max_value - Value to apply above threshold (255 = white) 
				#   object_type - 'light' (default) or 'dark'. If the object is lighter than 
				#                 the background then standard threshold is done. If the object 
				#                 is darker than the background then inverse thresholding is done. 
				s_thresh = pcv.threshold.binary(gray_img=s, threshold=70, max_value=255, object_type='light')
				# s_thresh = pcv.threshold.binary(gray_img=s, threshold=20, max_value=255, object_type='light')


				# Median Blur to clean noise 
				# Inputs: 
				#   gray_img - Grayscale image data 
				#   ksize - Kernel size (integer or tuple), (ksize, ksize) box if integer input,
				#           (n, m) box if tuple input 
				s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)


				# An alternative to using median_blur is gaussian_blur, which applies 
				# a gaussian blur filter to the image. Depending on the image, one 
				# technique may be more effective than others. 
				# Inputs:
				#   img - RGB or grayscale image data
				#   ksize - Tuple of kernel size
				#   sigma_x - Standard deviation in X direction; if 0 (default), 
				#            calculated from kernel size
				#   sigma_y - Standard deviation in Y direction; if sigmaY is 
				#            None (default), sigmaY is taken to equal sigmaX
				gaussian_img = pcv.gaussian_blur(img=s_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None)


				# Convert RGB to LAB and extract the blue channel ('b')
				# Input:
				#   rgb_img - RGB image data 
				#   channel- Split by 'l' (lightness), 'a' (green-magenta), or 'b' (blue-yellow) channel
				b = pcv.rgb2gray_lab(rgb_img=img, channel='l') #use either channel as l (if light is dominated in org image). else a or b (if color is dominated in org image)


				# Threshold the blue channel image 
				b_thresh = pcv.threshold.binary(gray_img=b, threshold=90, max_value=255, 
																																		object_type='dark')
				# b_thresh = pcv.threshold.binary(gray_img=b, threshold=120, max_value=255, 
				#                                 object_type='dark')


				# Join the threshold saturation and blue-yellow images with a logical or operation 
				# Inputs: 
				#   bin_img1 - Binary image data to be compared to bin_img2
				#   bin_img2 - Binary image data to be compared to bin_img1
				bs = pcv.logical_or(bin_img1=s_mblur, bin_img2=b_thresh)


				# Appy Mask (for VIS images, mask_color='white')
				# Inputs:
				#   img - RGB or grayscale image data 
				#   mask - Binary mask image data 
				#   mask_color - 'white' or 'black' 
				masked = pcv.apply_mask(img=img, mask=bs, mask_color='white')


				# Convert RGB to LAB and extract the Green-Magenta and Blue-Yellow channels
				masked_a = pcv.rgb2gray_lab(rgb_img=masked, channel='a')
				masked_b = pcv.rgb2gray_lab(rgb_img=masked, channel='b')


				# Threshold the green-magenta and blue images
				maskeda_thresh = pcv.threshold.binary(gray_img=masked_a, threshold=115, 
																																								max_value=255, object_type='dark')
				maskeda_thresh1 = pcv.threshold.binary(gray_img=masked_a, threshold=135, 
																																									max_value=255, object_type='light')
				maskedb_thresh = pcv.threshold.binary(gray_img=masked_b, threshold=128, 
																																								max_value=255, object_type='light')


				# Join the thresholded saturation and blue-yellow images (OR)
				ab1 = pcv.logical_or(bin_img1=maskeda_thresh, bin_img2=maskedb_thresh)
				ab = pcv.logical_or(bin_img1=maskeda_thresh1, bin_img2=ab1)


				# Fill small objects (reduce image noise) 
				# Inputs: 
				#   bin_img - Binary image data 
				#   size - Minimum object area size in pixels (must be an integer), and smaller objects will be filled
				ab_fill = pcv.fill(bin_img=ab, size=200)
				# ab_fill = ab


				# Apply mask (for VIS images, mask_color=white)
				masked2 = pcv.apply_mask(img=masked, mask=ab_fill, mask_color='white')


				# Identify objects
				# Inputs: 
				#   img - RGB or grayscale image data for plotting 
				#   mask - Binary mask used for detecting contours 
				id_objects, obj_hierarchy = pcv.find_objects(img=masked2, mask=ab_fill)



				#load detecor model and detect the ROI using detector
				with open(detector_label_pkl, 'rb') as classes: #path eg: r'D:\plant_proj\growth_prediction\labels.pickle',
						label = pickle.load(classes)

				model = core.Model.load(detector_model, label) #change the 'Model' class argument named as 'model_name' manually in original source code of core.py to the specific model name used while training. (path eg: 'D:/plant_proj/growth_prediction/model_weights.pth')
				torch_model = model.get_internal_model()

				read_img = utils.read_image(img_path)  # Helper function to read in images

				labels, boxes, scores = model.predict(read_img)  # Get all predictions on an image
				# predictions = model.predict_top(image)  #output gives: labels, boxes, scores
				predictions = model.predict(read_img)#output gives: labels, boxes, scores


				print(labels, boxes, scores)
				print(predictions)

				# visualize.show_labeled_image(img, predictions[1][0]) 

				x_cord, y_cord , width, height =  predictions[1][0]

				# x_cord, y_cord = co_ord_corners[0]
				# width, height = co_ord_corners[1]
				print(f"x-coordinate: {x_cord}, y-coordinate: {y_cord}, width: {width}, height: {height}")


				# Define the region of interest (ROI) 
				# Inputs: 
				#   img - RGB or grayscale image to plot the ROI on 
				#   x - The x-coordinate of the upper left corner of the rectangle 
				#   y - The y-coordinate of the upper left corner of the rectangle 
				#   h - The height of the rectangle 
				#   w - The width of the rectangle 
				roi1, roi_hierarchy= pcv.roi.rectangle(img=masked2, x=x_cord, y=y_cord, h=height-y_cord, w=width-x_cord)
				# roi1, roi_hierarchy= pcv.roi.rectangle(img=masked2, x=130, y=40, h=550, w=400)
				# roi1, roi_hierarchy= pcv.roi.rectangle(img=masked2, x=265, y=500, h=100, w=110)


				# Decide which objects to keep
				# Inputs:
				#    img            = img to display kept objects
				#    roi_contour    = contour of roi, output from any ROI function
				#    roi_hierarchy  = contour of roi, output from any ROI function
				#    object_contour = contours of objects, output from pcv.find_objects function
				#    obj_hierarchy  = hierarchy of objects, output from pcv.find_objects function
				#    roi_type       = 'partial' (default, for partially inside the ROI), 'cutto', or 
				#                     'largest' (keep only largest contour)
				roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img=img, roi_contour=roi1, 
																																																																	roi_hierarchy=roi_hierarchy, 
																																																																	object_contour=id_objects, 
																																																																	obj_hierarchy=obj_hierarchy,
																																																																	roi_type='partial')  #use 'roitype = cutto if cutto doesnt give proper mask'


				# Object combine kept objects
				# Inputs:
				#   img - RGB or grayscale image data for plotting 
				#   contours - Contour list 
				#   hierarchy - Contour hierarchy array 
				obj, mask = pcv.object_composition(img=img, contours=roi_objects, hierarchy=hierarchy3)


				############### Analysis ################ 
				# Find shape properties, data gets stored to an Outputs class automatically
				# Inputs:
				#   img - RGB or grayscale image data 
				#   obj- Single or grouped contour object
				#   mask - Binary image mask to use as mask for moments analysis 
				#   label - Optional label parameter, modifies the variable name of observations recorded. (default `label="default"`)filled_img = pcv.morphology.fill_segments(mask=cropped_mask, objects=edge_objects)
				analysis_image = pcv.analyze_object(img=img, obj=obj, mask=mask, label="default")


				# Access data stored out from analyze_object
				print("pcv.outputs.observations['default']:", pcv.outputs.observations['default'])

				plant_solidity = pcv.outputs.observations['default']['solidity']['value']
				print('plant_solidity: ', plant_solidity)


				# 19205/130378.5   #[step 1], #area/convex hull area =  solidity

				#Pixel value (total height or width of image in pixels) ÷ DPI value = image width in inches
				# (19205/120)   #[step 2],  #area/dpi (output is in square inches) #here in data, our image dpi value is 120

				# (12*12)   #[step 3],  #this means 1 square feet

				# total_sq_ft = (19205/120)/(12*12)   #[step 4],  #(total square inches)/(square inches in one square feet [which is 12*12=144]) , output is in square feet

				# 12*total_sq_ft    #[step 5], #formula: 12 inches * total sqaure feet = (output is height in inches)


				#calculate height
				total_sq_ft = (pcv.outputs.observations['default']['area']['value']/120)/(12*12) #(total square inches)/(square inches in one square feet [which is 12*12=144]) , output is in square feet
				len_height = 12*total_sq_ft #formula: 12 inches * total sqaure feet = (output is height in inches)
				print('Growth length/height is: ', len_height, 'inches')


				leaves_cnt_pcv = pcv.outputs.observations['default']['convex_hull_vertices']['value'] #count of total number of leaves/branches (count of convex hull vertices)
				print('Total approximate leaves using pcv approach- (check 1): ',leaves_cnt_pcv)


				contours, hierarchies= cv2.findContours(ab_fill.copy(), cv2.RETR_CCOMP,
						cv2.CHAIN_APPROX_SIMPLE)
				contours_poly = [None] * len(contours)
				# The Bounding Rectangles will be stored here:
				boundRect = []
				# Just look for the outer bounding boxes:
				for i, c in enumerate(contours):
						if hierarchies[0][i][3] == -1:
										contours_poly[i] = cv2.approxPolyDP(c, 3, True)
										boundRect.append(cv2.boundingRect(contours_poly[i]))
				# Draw the bounding boxes on the (copied) input image:
				for i in range(len(boundRect)):
						color = (0, 255, 0)
						cv2.rectangle(img, (int(boundRect[i][0]), int(boundRect[i][1])), \
																(int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)

				leaves_cnt_contours = len(boundRect)

				print('Total approximate leaves using contours approach- (check 2): ', leaves_cnt_contours)

				print('len_height',len_height)
				print('leaves_cnt_pcv',leaves_cnt_pcv)
				print('leaves_cnt_contours',leaves_cnt_contours)

				print('=====', single_path.split('\\')[-1])
				pred_values_len_height[single_path.split('\\')[-1]] = len_height #store all image height pred name and value inside each subfolder
				pred_values_leaf_cnt_countour[single_path.split('\\')[-1]] = leaves_cnt_pcv #store all image leaf count using plantcv pred name and value inside each subfolder
				pred_values_leaf_cnt_pvc[single_path.split('\\')[-1]] = leaves_cnt_contours #store all image leaf count using contours pred name and value inside each subfolder
				
			all_preds_len_height[dir_name] = pred_values_len_height #store pred_values_sub_fol subfolders name wise
			all_preds_leaf_cnt_contour[dir_name] = pred_values_leaf_cnt_countour #store pred_values_sub_fol subfolders name wise
			all_preds_leaf_cnt_pvc[dir_name] = pred_values_leaf_cnt_pvc #store pred_values_sub_fol subfolders name wise

		print(all_preds_len_height)
		print(all_preds_leaf_cnt_contour)
		print(all_preds_leaf_cnt_pvc)

		len_output_all_data = len(output_all_data)
		len_output_new_data = len(output_new_data)
		# generate few featured attributes to insert the value in already saved values in DB of predicted outcome of plant data
		for ind, ech_plant in enumerate(output_new_data):
			len_output_all_data = len_output_all_data
			len_output_new_data = len_output_new_data
			if ind == 0:
				len_output_all_data = (len_output_all_data - len_output_new_data) + 1
			else:
				len_output_all_data = len_output_all_data + 1

			conn = pymysql.connect( 
				host='localhost', 
				user='root', 
				db='plant', 
				password='1234',
				port=3307
				) 

			conn.begin()
			cur = conn.cursor() 
			print(ech_plant)
			exec_statement = ' UPDATE growth_plant SET id= {class_name} , results_len_height= "{output_res_height}", results_leaves_cnt_pcv= "{output_res_cnt_pcv}",results_leaves_cnt_contours= "{output_res_cnt_contour}", processed_new = "{predict_old_new}", status = "{status_db}"  WHERE folder_name = "{fold_name}"'.format(class_name = len_output_all_data, output_res_height = all_preds_len_height[list(all_preds_len_height.keys())[ind]], output_res_cnt_contour = all_preds_leaf_cnt_contour[list(all_preds_leaf_cnt_contour.keys())[ind]], output_res_cnt_pcv = all_preds_leaf_cnt_pvc[list(all_preds_leaf_cnt_pvc.keys())[ind]], fold_name = ech_plant[1], predict_old_new = 'processed', status_db = 'success')
			print(exec_statement)
			cur.execute(exec_statement)
			conn.commit() 

		# conn.rollback() #if you want to go back to previous values in table
		conn.close()    
		print('database updated!!---')

		answer = 'success' #success means atleast one result found while predicting
		print(answer)

	except:

		answer = 'error'
		print(answer)

	return jsonify(answer)



if __name__ == '__main__':

	app.run(host='127.0.0.1', port=5000, debug=True)

	growth_stage() #function call
	
	#pass paramters value in url on browser after running app like below
	# http://127.0.0.1:5000/growth_pred?detector_label_pkl=D:\plant_proj\growth_prediction\labels.pickle&detector_model=D:/plant_proj/growth_prediction/model_weights.pth

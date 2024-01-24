import numpy as np
import os
import cv2,shutil
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys
from sys import exit

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense,BatchNormalization
from keras.layers import Input,Activation
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model, Input,load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv2D
from keras.regularizers import l2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras.layers import Input,Activation
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.regularizers import l2

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from flask import Flask, render_template, Response
# from flask import redirect, url_for

import pymysql 




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


FILE_PATH = os.path.dirname(os.path.realpath(__file__))
print(FILE_PATH)



# * ---------- Create App --------- *
app = Flask(__name__)
CORS(app, support_credentials=True)



# @app.route('/')
# def index():
# 	return render_template('index.html')


def custom_model(inputShape, trainBool, num_class):
	init = "he_normal"
	reg=l2(0.0005)
	chanDim = -1
	classes = num_class
	inputShape = inputShape
	model = Sequential()
	model.add(Conv2D(16, (7, 7), strides=(2, 2), padding="valid",kernel_initializer=init, kernel_regularizer=reg,input_shape=inputShape,trainable = trainBool))
			# here we stack two CONV layers on top of each other where
			# each layers will learn a total of 32 (3x3) filters
	model.add(Conv2D(32, (3, 3), padding="same",
		kernel_initializer=init, kernel_regularizer=reg,trainable = trainBool))
	model.add(Activation("relu",trainable = trainBool))
	model.add(BatchNormalization(axis=chanDim,trainable = trainBool))
	model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
		kernel_initializer=init, kernel_regularizer=reg,trainable = trainBool))
	model.add(Activation("relu",trainable = trainBool))
	model.add(BatchNormalization(axis=chanDim,trainable = trainBool))
	model.add(Dropout(0.25,trainable = trainBool))
	model.add(Conv2D(64, (3, 3), padding="same",
				kernel_initializer=init, kernel_regularizer=reg,trainable = trainBool))
	model.add(Activation("relu",trainable = trainBool))
	model.add(BatchNormalization(axis=chanDim,trainable = trainBool))
	model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same",
		kernel_initializer=init, kernel_regularizer=reg,trainable = trainBool))
	model.add(Activation("relu",trainable = trainBool))
	model.add(BatchNormalization(axis=chanDim,trainable = trainBool))
	model.add(Dropout(0.25,trainable = trainBool))
	# increase the number of filters again, this time to 128
	model.add(Conv2D(128, (3, 3), padding="same",
		kernel_initializer=init, kernel_regularizer=reg,trainable = trainBool))
	model.add(Activation("relu",trainable = trainBool))
	model.add(BatchNormalization(axis=chanDim,trainable = trainBool))
	model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same",
		kernel_initializer=init, kernel_regularizer=reg,trainable = trainBool))
	model.add(Activation("relu",trainable = trainBool))
	model.add(BatchNormalization(axis=chanDim,trainable = trainBool))
	model.add(Dropout(0.25,trainable = trainBool))
	
	model.add(Flatten()) ### 7
	model.add(Dense(512, kernel_initializer=init))  ### 6
	model.add(Activation("relu"))  ##5
	model.add(BatchNormalization())  ###4
	model.add(Dropout(0.5))   ### 3
	 
	model.add(Dense(classes))  ##2
	model.add(Activation("softmax"))  ##1

	return model



def img_to_encoding(image_path, model_cattle):

	print('--------image to encode--------')
   # print(image_path)
	print(image_path)
	img = cv2.imread(image_path)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#print("grayshape",gray.shape)
	resized = cv2.resize(gray, (224,224))
	resized = resized * (1./255)
	#print(resized.shape)

	gray = np.reshape(resized,[1,224,224,1])  #### starting 1 refers to number of examples

	classes = model_cattle.predict(gray)
	print('--test image encoded--')

	return classes






#*--------------------------- Function calls from server --------------------------------

@app.route('/retrainmodelencode', methods=['GET','POST'])
def retrain_new_data_gen_encode():
	# json_data = request.get_json()

    # To connect MySQL database 
	conn = pymysql.connect( 
		host='localhost', 
		user='username', 
		db='DB name', 
		password='password',
		port='port number'
	    ) 
	  
	cur = conn.cursor() 
	cur.execute("select * from disease_plant where registered_new_proc = 'new'") 
	output_new_data = cur.fetchall() 
	# folder_name = output[0][1]
	# train_data_path = output[0][2]

	print('output_new_data::: ', output_new_data)

	# To close the connection 
	conn.close() 
  
    source_path = output_new_data[0][2]
 
	try:

		# a = request.args.get('param1', None)
		# b = request.args.get('param2', None)

		INIT_LR = 1e-3
		EPOCHS = 2
		BS = 8

		# try:
		# 	del known_dir_paths
		# except:
		# 	pass

		known_dir_paths = []

		# for (dirpath, dirnames, filenames) in os.walk('{path}/Data/Crop1-tomato/train/'.format(path=FILE_PATH)): #change path as per the server
		# 	known_dir_paths.append(dirpath)

		for (dirpath, dirnames, filenames) in os.walk('{path}'.format(path=source_path)): #change path as per the server
			known_dir_paths.append(dirpath)

		classes = int(len(known_dir_paths[1:])) ### ------- adding more classes
		print('Total classes:', classes)

		data_path = '{path}'.format(path=source_path) #change path as per the server 
		# if we dont have separate validation data then use below, else use the commented line which comes after below line
		data_path_valid = '{path}'.format(path=source_path) # change path as per the server
		# data_path_valid = '{path}/Data/Crop1-tomato/valid/'.format(path=FILE_PATH) # change path as per the server
		# data_path_test = '{path}/Data/Crop1-tomato/test/'.format(path=FILE_PATH) # change path as per the server


		train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=15,fill_mode="nearest")
		valid_datagen = ImageDataGenerator(rescale=1./255)
		# test_datagen = ImageDataGenerator(rescale=1./255)

		train_generator = train_datagen.flow_from_directory(
			directory=data_path,
			target_size=(224, 224),
			color_mode="grayscale",
			batch_size=BS,
			class_mode="categorical",
			shuffle=True,
			seed=42
		)

		valid_generator = valid_datagen.flow_from_directory(
			directory=data_path_valid,
			target_size=(224, 224),
			color_mode="grayscale",
			batch_size=BS,
			class_mode="categorical",
			shuffle=True,
			seed=42
		)

		# test_generator = test_datagen.flow_from_directory(
		#     directory=data_path_valid,
		#     target_size=(224, 224),
		#     color_mode="grayscale",
		#     batch_size=1,
		#     class_mode='categorical',
		#     shuffle=False,
		#     seed=42
		# )


		STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
		STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
		# STEP_SIZE_TEST=test_generator.n//test_generator.batch_size






		old_classes = classes - len(output_new_data)
		print('Total Old classes:', old_classes)

		old_classes = classes


		baseModel = custom_model((224,224,1), False, old_classes)  ### grayscale input
		pretrained_weights = '{path}/model_best_weights_plant.h5'.format(path=FILE_PATH)   #######Loading 33 class model weights
		baseModel.load_weights(pretrained_weights)

		baseModel.pop()
		baseModel.pop()
		baseModel.pop()
		baseModel.pop()
		baseModel.pop()
		baseModel.pop()
		baseModel.pop()

		# print(baseModel.summary())





		headModel = baseModel

		init = "he_normal"
		headModel.add(Flatten())
		headModel.add(Dense(512, kernel_initializer=init))
		headModel.add(Activation("relu"))
		headModel.add(BatchNormalization())
		headModel.add(Dropout(0.5))
		# softmax classifier
		headModel.add(Dense(classes))     ###### added 27 classes more total 60 classes now
		headModel.add(Activation("softmax"))

		# print(headModel.summary())




		print("[INFO] compiling model...")
		opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
		headModel.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

		checkpoint = ModelCheckpoint('{path}/model_best_weights_plant.h5'.format(path=FILE_PATH), monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)

		# train the head of the network
		print("[INFO] training head...")
		H = headModel.fit_generator(train_generator,steps_per_epoch=STEP_SIZE_TRAIN,validation_data=valid_generator,validation_steps=STEP_SIZE_VALID,epochs=EPOCHS,callbacks = [checkpoint])

		print('### --- model re-trained on new data...')
		# print(headModel.summary())





		#----------------- generatng pickle file of probs of images--------------------------

		# data_path_train_cattle = '{path}/Data/Crop1-tomato/train/'.format(path=FILE_PATH) #change path as per the server 
		data_path_train_cattle = '{path}'.format(path=source_path) #change path as per the server 
		model_cattle = load_model('{path}/model_best_weights_plant.h5'.format(path=FILE_PATH)) #change path as per the server 
		database = {}


		all_subfolders = []

		# directory_path = '{path}\\Data\\Crop1-tomato\\train\\'.format(path=FILE_PATH)
		directory_path = '{path}'.format(path=source_path)

		for i in os.listdir(directory_path):
			path = directory_path + i
			all_subfolders.append(path)

		all_image_path = []
		for subdir in all_subfolders:
			for (dirpath, dirnames, filenames) in os.walk(subdir): #change path as per the server
				for img_name in filenames:

					image_names = dirpath + '\\' + img_name
					all_image_path.append(image_names)

		
		for ind, single_path in enumerate(all_image_path):    
			sub_folder_name = single_path.split('\\')[-2]
			sub_folder_name = sub_folder_name + '=' +str(ind) 
			database[sub_folder_name] = img_to_encoding(single_path, model_cattle)

		print(len(database))

		outfile = open('{path}/plant_images_prob_encodings.pkl'.format(path=FILE_PATH),'wb') #change path as per the server 
		pickle.dump(database, outfile)
		outfile.close()

		print('### --- saved pickle successfully for plant image probabilities...')

		# generate few featured attributes to insert the value in already saved values in DB of newly trained plant data which was queried at the start
		for ech_plant in output_new_data:
			old_classes = old_classes + 1
			status_reg = 'success'

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

			exec_statement = " UPDATE disease_plant SET id= {class_name} , status= '{status}' , registered_new_proc = '{reg_old_new}' WHERE folder_name = '{fold_name}'".format(class_name = int(old_classes), status = status_reg, fold_name = ech_plant[1], reg_old_new = 'processed')
			print(exec_statement)

			cur.execute(exec_statement)

			# conn.rollback() #if you want to go back to previous values in table
			conn.commit() #commit and confirm the update in table of DB
			conn.close()  			# To close the connection 

			print('Values updated in database table successfully!!')
			answer = 'success'
			print(answer)


	except:

		answer = 'error'
		print(answer)


	# Return the data to the front
	return jsonify(answer)






#*--------------------------- predict/test query image and return the name of plant (label)-----------------------------------


@app.route('/predqueryimage',methods=['GET','POST'])
# @cross_origin(supports_credentials=True)

# def testqueryimage(id, folder_name_image):
def pred_query_image():

	    # To connect MySQL database 
	conn = pymysql.connect( 
	    host='localhost', 
	    user='root', 
	    db='plant', 
	    password='1234',
	    port=3307
	    ) 
	  
	cur = conn.cursor() 
	cur.execute("select * from disease_plant_pred where pred_proc_new = 'new'") 
	output_new_data = cur.fetchall() 
	# folder_name = output[0][1]
	# train_data_path = output[0][2]

	cur.execute("select * from disease_plant_pred") 
	output_all_data = cur.fetchall() 

	print('output_new_data::: ', len(output_new_data))

	print('len(output_all_data)::: ', len(output_all_data))

	# To close the connection 
	conn.close() 

    source_path = output_new_data[0][2]


	try:

		# input_id = str(id)
		# print(input_id)
		# folder_name_image = str(folder_name_image)
		# print(folder_name_image)

		infile = open('{path}/plant_images_prob_encodings.pkl'.format(path=FILE_PATH),'rb')
		# infile = open('{path}\\exe_image_encodings\\dist\\'.format(path=FILE_PATH)+'encodings.pkl','rb')
		imported_data = pickle.load(infile)
		infile.close()



		global answer_test

		model_cattle = load_model('{path}/model_best_weights_plant.h5'.format(path=FILE_PATH)) #change path as per the server 

		all_subfolders = []
		# image_path = '{path}\\test\\'.format(path=FILE_PATH) #testimages[i]----- (query image) change path as per server
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

		all_preds = {}		

		for ind, dir_name in enumerate(sub_fol_image_path_dict.keys()): 

			pred_values_sub_fol = {}

			for ind, single_path in enumerate(sub_fol_image_path_dict[dir_name]):

				yes_matched = {}

			#     print("testimage, databaseimage:",image_path,identity)
				encoding = img_to_encoding(single_path, model_cattle)

				for i in imported_data.keys():
					
					identity = str(i)

					dist = np.linalg.norm(encoding-imported_data[identity] )
					if dist < 0.2 :
						# print(identity)
				#         print("YES, It is plant with ID "+str(identity.split('_')[0]))
						# yes_matched.append(int(identity.split('_')[0]))
						yes_matched[str(identity.split('JPG')[0])] = str(dist)

						# break

					else:
				#         print("No, it is not plant with ID "+str(identity.split('_')[0]))
						pass

				sorted_dict = dict(sorted(yes_matched.items(), key=lambda item: item[1]))
				sorted_dict_keys = list(sorted_dict.keys())[0]
				sorted_dict_values = list(sorted_dict.values())[0]

				yes_matched = {}
				yes_matched[str(sorted_dict_keys) ]= sorted_dict_values

				#     print(dist)

				# unique_ids = pd.Series(yes_matched).unique()

				# answer_test = {}

				# for ind, val in enumerate(unique_ids):
				# 	answer_test[int(ind)] = str(val)

				# answer_test = yes_matched
				print('yes_matched object value: ', yes_matched)
				yes_matched_key = "".join(list(yes_matched.keys()))
				yes_matched_key = yes_matched_key.split('=')[0] + '=' + str(ind)

				# id_folder = single_path.split('/')[-2] #use this only if to use below condition

				#use below only if we should return the predicted plant name by comparing between query and saved image name and matched plant name
				# if id_folder == yes_matched_key:
				# 	# answer_test = {}
				# 	# indexed = np.where(np.array(list(yes_matched.keys())) == id_folder)
				# 	# print('index: ',indexed)
				# 	# id_val = int(np.asarray([indexed]))
				# 	# print('id_val: ',id_val)
				# 	# answer_test[list(yes_matched.keys())[id_val]] = list(yes_matched.values())[id_val]
				# 	# print('answer_test: ',answer_test)
				# 	print('Match found, predicted is: {} and original is: {}'.format(yes_matched_key, id_folder) )
				# 	answer_test = 'success'
				# 	print( answer_test)
				# else:
				# 	print('no match found! predicted is: {} and original is: {}'.format(yes_matched_key, id_folder))
				# 	answer_test = 'no success'
				# 	print( answer_test)
					
				# input_folder_img = single_path.split('/')[-2]

				pred_values_sub_fol[yes_matched_key] = sorted_dict_values #store all image pred name and value inside each subfolder
				# all_preds[yes_matched_key] = sorted_dict_values #store resulting matched plant name and distance compare value for all query images
			all_preds[dir_name] = pred_values_sub_fol #store pred_values_sub_fol subfolders name wise
		print(all_preds)

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
			exec_statement = ' UPDATE disease_plant_pred SET id= {class_name} , results_pred= "{output_res}", pred_proc_new = "{predict_old_new}"  WHERE folder_name = "{fold_name}"'.format(class_name = len_output_all_data, output_res = all_preds[list(all_preds.keys())[ind]], fold_name = ech_plant[1], predict_old_new = 'processed')
			print(exec_statement)
			cur.execute(exec_statement)
			conn.commit() 

		# conn.rollback() #if you want to go back to previous values in table
		conn.close()  	
		print('database updated!!---')

		answer_test = 'success' #success means atleast one result found while predicting
		print(answer_test)
	except:

		answer_test = 'error' #error means no result found while predicting, based on mentioned distance compare threshold
		print(answer_test)


	return jsonify(answer_test)
	# return answer_test





# # * -------------------- RUN SERVER -------------------- *

if __name__ == '__main__':
	# * --- DEBUG MODE: --- *
	# app.run(host='localhost', port=5000, debug=True)
	app.run(host='127.0.0.1', port=5000, debug=True)
	#  * --- DOCKER PRODUCTION MODE: --- *
	# app.run(host='0.0.0.0', port=os.environ['PORT']) -> DOCKER
	# pred_query_image(sys.argv[1], sys.argv[2])
	# generate_encoding()
	# retrain_new_data_gen_encode()
	pred_query_image()


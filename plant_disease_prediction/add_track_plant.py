import numpy as np
import os
import cv2,shutil
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys
from sys import exit

# from keras.preprocessing.image import ImageDataGenerator
# from keras.applications import VGG16
# from keras.layers import AveragePooling2D
# from keras.layers import Dropout
# from keras.layers import Flatten
# from keras.layers import Dense,BatchNormalization
# from keras.layers import Input,Activation
# from keras.models import Model,Sequential
# from keras.optimizers import Adam
# from keras.utils import to_categorical
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from keras.optimizers import Adam, SGD, RMSprop
# from keras.models import Model, Input,load_model
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.layers.convolutional import Conv2D
# from keras.regularizers import l2

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.layers import AveragePooling2D
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dense,BatchNormalization
# from tensorflow.keras.layers import Input,Activation
# from tensorflow.keras.models import Model,Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import Model,load_model
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.regularizers import l2

# from flask import Flask, request, jsonify
# from flask_cors import CORS, cross_origin
# from flask import Flask, render_template, Response


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


FILE_PATH = os.path.dirname(os.path.realpath(__file__))
print(FILE_PATH)

# app = Flask(__name__)
# CORS(app, support_credentials=True)




# def custom_model(inputShape, trainBool, num_class):
# 	init = "he_normal"
# 	reg=l2(0.0005)
# 	chanDim = -1
# 	classes = num_class
# 	inputShape = inputShape
# 	model = Sequential()
# 	model.add(Conv2D(16, (7, 7), strides=(2, 2), padding="valid",kernel_initializer=init, kernel_regularizer=reg,input_shape=inputShape,trainable = trainBool))
# 			# here we stack two CONV layers on top of each other where
# 			# each layerswill learn a total of 32 (3x3) filters
# 	model.add(Conv2D(32, (3, 3), padding="same",
# 		kernel_initializer=init, kernel_regularizer=reg,trainable = trainBool))
# 	model.add(Activation("relu",trainable = trainBool))
# 	model.add(BatchNormalization(axis=chanDim,trainable = trainBool))
# 	model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
# 		kernel_initializer=init, kernel_regularizer=reg,trainable = trainBool))
# 	model.add(Activation("relu",trainable = trainBool))
# 	model.add(BatchNormalization(axis=chanDim,trainable = trainBool))
# 	model.add(Dropout(0.25,trainable = trainBool))
# 	model.add(Conv2D(64, (3, 3), padding="same",
# 				kernel_initializer=init, kernel_regularizer=reg,trainable = trainBool))
# 	model.add(Activation("relu",trainable = trainBool))
# 	model.add(BatchNormalization(axis=chanDim,trainable = trainBool))
# 	model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same",
# 		kernel_initializer=init, kernel_regularizer=reg,trainable = trainBool))
# 	model.add(Activation("relu",trainable = trainBool))
# 	model.add(BatchNormalization(axis=chanDim,trainable = trainBool))
# 	model.add(Dropout(0.25,trainable = trainBool))
# 	# increase the number of filters again, this time to 128
# 	model.add(Conv2D(128, (3, 3), padding="same",
# 		kernel_initializer=init, kernel_regularizer=reg,trainable = trainBool))
# 	model.add(Activation("relu",trainable = trainBool))
# 	model.add(BatchNormalization(axis=chanDim,trainable = trainBool))
# 	model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same",
# 		kernel_initializer=init, kernel_regularizer=reg,trainable = trainBool))
# 	model.add(Activation("relu",trainable = trainBool))
# 	model.add(BatchNormalization(axis=chanDim,trainable = trainBool))
# 	model.add(Dropout(0.25,trainable = trainBool))
	
# 	model.add(Flatten()) ### 7
# 	model.add(Dense(512, kernel_initializer=init))  ### 6
# 	model.add(Activation("relu"))  ##5
# 	model.add(BatchNormalization())  ###4
# 	model.add(Dropout(0.5))   ### 3
	 
# 	model.add(Dense(classes))  ##2
# 	model.add(Activation("softmax"))  ##1

# 	return model



def img_to_encoding(image_path,model_cattle):
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
	print('test image encoded')
	return classes






#*--------------------------- convert base64 image to pil and save all images in folder --------------------------------


# @app.route('/saveimagetrain')
# # @cross_origin(supports_credentials=True)

# def saveimagetrain():



#*---------------------------------------  transfer learning code block -------------------------------------------
# try:
# 	del known_dir_paths
# except:
# 	pass


# @app.route('/transferlearn')
# @cross_origin(supports_credentials=True)

def generate_encoding():
	# json_data = request.get_json()
	try:

		# a = request.args.get('param1', None)
		# b = request.args.get('param2', None)

		# INIT_LR = 1e-3
		# EPOCHS = 2
		# BS = 8

		#/cattle_model_weights/plant_disease_prediction/Data/cattle_dataset1/train/'

		# try:
		# 	del known_dir_paths
		# except:
		# 	pass

		# known_dir_paths = []

		# for (dirpath, dirnames, filenames) in os.walk('{path}/Data/cattle_dataset1/train/'.format(path=FILE_PATH)): #change path as per the server
		# 	known_dir_paths.append(dirpath)


		# classes = int(len(known_dir_paths[1:])) ### ------- adding more classes

		# data_path = '{path}/Data/cattle_dataset1/train/'.format(path=FILE_PATH) #change path as per the server 
		# data_path_valid = '{path}/Data/cattle_dataset1/valid/'.format(path=FILE_PATH) # change path as per the server
		# # data_path_test = '{path}/cattle_model_weights/plant_disease_prediction/Data/cattle_dataset1/test/'.format(path=FILE_PATH) # change path as per the server


		# train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=15,fill_mode="nearest")
		# valid_datagen = ImageDataGenerator(rescale=1./255)
		# # test_datagen = ImageDataGenerator(rescale=1./255)

		# train_generator = train_datagen.flow_from_directory(
		# 	directory=data_path,
		# 	target_size=(224, 224),
		# 	color_mode="grayscale",
		# 	batch_size=BS,
		# 	class_mode="categorical",
		# 	shuffle=True,
		# 	seed=42
		# )

		# valid_generator = valid_datagen.flow_from_directory(
		# 	directory=data_path_valid,
		# 	target_size=(224, 224),
		# 	color_mode="grayscale",
		# 	batch_size=BS,
		# 	class_mode="categorical",
		# 	shuffle=True,
		# 	seed=42
		# )

		# # test_generator = test_datagen.flow_from_directory(
		# #     directory=data_path_valid,
		# #     target_size=(224, 224),
		# #     color_mode="grayscale",
		# #     batch_size=1,
		# #     class_mode='categorical',
		# #     shuffle=False,
		# #     seed=42
		# # )


		# STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
		# STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
		# # STEP_SIZE_TEST=test_generator.n//test_generator.batch_size






		# old_classes = classes - 1
		# # old_classes = classes


		# baseModel = custom_model((224,224,1), False, old_classes)  ### grayscale input
		# pretrained_weights = '{path}/model_best_weights_plant.h5'.format(path=FILE_PATH)   #######Loading 33 class model weights
		# baseModel.load_weights(pretrained_weights)

		# baseModel.pop()
		# baseModel.pop()
		# baseModel.pop()
		# baseModel.pop()
		# baseModel.pop()
		# baseModel.pop()
		# baseModel.pop()

		# # print(baseModel.summary())





		# headModel = baseModel

		# init = "he_normal"
		# headModel.add(Flatten())
		# headModel.add(Dense(512, kernel_initializer=init))
		# headModel.add(Activation("relu"))
		# headModel.add(BatchNormalization())
		# headModel.add(Dropout(0.5))
		# # softmax classifier
		# headModel.add(Dense(classes))     ###### added 27 classes more total 60 classes now
		# headModel.add(Activation("softmax"))

		# # print(headModel.summary())




		# print("[INFO] compiling model...")
		# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
		# headModel.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

		# checkpoint = ModelCheckpoint('{path}/model_best_weights_plant.h5'.format(path=FILE_PATH), monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)

		# # train the head of the network
		# print("[INFO] training head...")
		# H = headModel.fit_generator(train_generator,steps_per_epoch=STEP_SIZE_TRAIN,validation_data=valid_generator,validation_steps=STEP_SIZE_VALID,epochs=EPOCHS,callbacks = [checkpoint])

		# print('### --- model re-trained on new data...')
		# # print(headModel.summary())





		#----------------- generatng pickle file of probs of images--------------------------

		# data_path_train_cattle = '{path}/Data/Crop1-tomato/all_train/'.format(path=FILE_PATH) #change path as per the server 
		data_path_train_cattle = '{path}/Data/Crop1-tomato/train/'.format(path=FILE_PATH) #change path as per the server 
		model_cattle = load_model('{path}/model_best_weights_plant.h5'.format(path=FILE_PATH)) #change path as per the server 
		database = {}


		all_subfolders = []

		directory_path = '{path}\\Data\\Crop1-tomato\\train\\'.format(path=FILE_PATH)
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


		answer = 'success'
		print(answer)

	except:

		answer = 'error'
		print(answer)


	return answer






#*--------------------------- predict/test query image and return the name of cattle (label)-----------------------------------


# @app.route('/testqueryimage/<int:id>', methods=['GET'])
# @cross_origin(supports_credentials=True)


def testqueryimage(id, folder_name_image):
	try:

		input_id = str(id)
		print(input_id)
		folder_name_image = str(folder_name_image)
		print(folder_name_image)

		infile = open('{path}/plant_images_prob_encodings.pkl'.format(path=FILE_PATH),'rb')
		# infile = open('{path}\\exe_image_encodings\\dist\\'.format(path=FILE_PATH)+'encodings.pkl','rb')
		imported_data = pickle.load(infile)
		infile.close()


		yes_matched = {}

		global answer_test

		model_cattle = load_model('{path}/model_best_weights_plant.h5'.format(path=FILE_PATH)) #change path as per the server 
		
		image_path = '{path}/Data/Crop1-tomato/test/{image_folder}/{test}'.format(path=FILE_PATH, test=input_id, image_folder=folder_name_image) #testimages[i]----- (query image) change path as per server
		#     print("testimage, databaseimage:",image_path,identity)
		encoding = img_to_encoding(image_path, model_cattle)

		for i in imported_data.keys():
			
			identity = str(i)
		   # image_path = './cattlechan/dataindir/test/cownum35/35_6.jpg'

			dist = np.linalg.norm(encoding-imported_data[identity] )
			if dist < 0.2 :
				# print(identity)
		#         print("YES, It is cow with ID "+str(identity.split('_')[0]))
				# yes_matched.append(int(identity.split('_')[0]))
				yes_matched[str(identity.split('JPG')[0])] = str(dist)

				# break

			else:
		#         print("No, it is not cow with ID "+str(identity.split('_')[0]))
				pass

		sorted_dict = dict(sorted(yes_matched.items(), key=lambda item: item[1]))
		print(sorted_dict)
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
		yes_matched_key = yes_matched_key.split('=')[0]
		id_folder = image_path.split('/')[-2]
		if id_folder == yes_matched_key:
			# answer_test = {}
			# indexed = np.where(np.array(list(yes_matched.keys())) == id_folder)
			# print('index: ',indexed)
			# id_val = int(np.asarray([indexed]))
			# print('id_val: ',id_val)
			# answer_test[list(yes_matched.keys())[id_val]] = list(yes_matched.values())[id_val]
			# print('answer_test: ',answer_test)
			print('Match found, predicted is: {} and original is: {}'.format(yes_matched_key, id_folder) )
			answer_test = 'success'
			print( answer_test)
		else:
			print('no match found! predicted is: {} and original is: {}'.format(yes_matched_key, id_folder))
			answer_test = 'no success'
			print( answer_test)
			
		input_folder_img = image_path.split('/')[-2]

	except:

		answer_test = 'error'
		print(answer_test)


	return answer_test





# # * -------------------- RUN SERVER -------------------- *

if __name__ == '__main__':
	# * --- DEBUG MODE: --- *
	# app.run(host='localhost', port=5000, debug=True)
	# app.run(host='127.0.0.1', port=5000, debug=True)
	#  * --- DOCKER PRODUCTION MODE: --- *
	# app.run(host='0.0.0.0', port=os.environ['PORT']) -> DOCKER
	testqueryimage(sys.argv[1], sys.argv[2])
	# generate_encoding()



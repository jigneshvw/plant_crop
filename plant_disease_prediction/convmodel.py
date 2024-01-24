import numpy as np
import os
import cv2,shutil
import matplotlib.pyplot as plt
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


INIT_LR = 1e-3
EPOCHS = 10
BS = 15
classes = 15

data_path = os.getcwd()+'/Data/Crop1-tomato/train/'
data_path_valid = os.getcwd()+'/Data/Crop1-tomato/valid/'
data_path_test = os.getcwd()+'//Data/Crop1-tomato/test/'


train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=15,fill_mode="nearest")
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

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


test_generator = test_datagen.flow_from_directory(
    directory=data_path_valid,
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=1,
    class_mode='categorical',
    shuffle=False,
    seed=42
)


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size



# place the head FC model on top of the base model (this will become
# the actual model we will train)
init = "he_normal"
reg=l2(0.0005)
chanDim = -1
inputShape = (224,224,1)
model = Sequential()
model.add(Conv2D(16, (7, 7), strides=(2, 2), padding="valid",kernel_initializer=init, kernel_regularizer=reg,input_shape=inputShape))
		# here we stack two CONV layers on top of each other where
		# each layerswill learn a total of 32 (3x3) filters
model.add(Conv2D(32, (3, 3), padding="same",
	kernel_initializer=init, kernel_regularizer=reg))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
	kernel_initializer=init, kernel_regularizer=reg))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same",
			kernel_initializer=init, kernel_regularizer=reg))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same",
	kernel_initializer=init, kernel_regularizer=reg))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Dropout(0.25))
# increase the number of filters again, this time to 128
model.add(Conv2D(128, (3, 3), padding="same",
	kernel_initializer=init, kernel_regularizer=reg))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same",
	kernel_initializer=init, kernel_regularizer=reg))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, kernel_initializer=init))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# softmax classifier
model.add(Dense(classes))
model.add(Activation("softmax"))


#model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

checkpoint = ModelCheckpoint('model_best_weights_plant.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)

# train the head of the network
print("[INFO] training head...")
H = model.fit_generator(train_generator,steps_per_epoch=STEP_SIZE_TRAIN,validation_data=valid_generator,validation_steps=STEP_SIZE_VALID,epochs=EPOCHS,callbacks = [checkpoint])

print(model.summary())
model.save('plantclass_vgg.hdf5')

print("validation accuracy")

print(model.evaluate_generator(generator=valid_generator,steps=STEP_SIZE_VALID))

print("test accuracy")
print(model.evaluate_generator(generator=test_generator,steps=STEP_SIZE_TEST))

print('----------Training completed--------------')
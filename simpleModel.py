'''
Code taken from the example found https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/.
Adapted by Ziyue Wang for Y4 project. Created on 14/1/2019
'''

import numpy
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend as K

#Writing a seed for reproducibility
seed = 1
numpy.random.seed(seed)

#Load data from input, gotta write something for this. block

train_data_dir = '128img/train'
validation_data_dir = '128img/train'
nb_train_samples = 1400
nb_validation_samples=600
epochs=50
batch_size=16				#Reduce this is we see problems. If using bluebear, might be smart to increase this. At home, use 128 max.


img_width, img_height = 128, 128
#input_shape=						#Input shape only needed for Conv

num_pixels = img_width*img_height
num_classes = 9 				#9 different categories for the output, this is tempoary workaround



#Let's define the model (simple) 

def simple_model():
	#create model
        model = Sequential()
        model.add(Dense(num_pixels,			#dimensionality of output space
			input_shape=(128,128,1),
			kernel_initializer='normal'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(9, kernel_initializer='normal', activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        return model



#After data is inputed, we should augment the data in some way.
train_datagen = ImageDataGenerator(
	rescale=1./255,					#Normalized inputs from 0-255 to 0-1
	zoom_range=0.2,		
	horizontal_flip=True,
	vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#Images should be inputed the same way as the code found in main to avoid confusion
train_generator = train_datagen.flow_from_directory(
	validation_data_dir,
        color_mode='grayscale',	
	target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='categorical',
	shuffle=True)

validation_generator = test_datagen.flow_from_directory(
	validation_data_dir,  
        color_mode='grayscale',
        target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='categorical',
	shuffle=True)

#build the model
model = simple_model()

model.fit_generator(
	train_generator,
	steps_per_epoch=nb_train_samples // batch_size,
	epochs=epochs,
	validation_data=validation_generator,
	validation_steps=nb_validation_samples // batch_size)

model.save_weights('very_simple.h5')
K.clear_session()


'''
A need to flatten the image for multi-layer perveptron models
flatten 128*128 pixel images.
Since this is taken off the example of using MINST number classification, they use the MINST data which has the labels as their first array component [0]
Need to find workaround for this. Possible just use the model part of the example?
'''




			

















   

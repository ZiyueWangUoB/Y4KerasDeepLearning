'''
Code taken from the example found https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/.
Adapted by Ziyue Wang for Y4 project. Created on 14/1/2019
'''

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications 

#Writing a seed for reproducibility
seed = 1
numpy.random.seed(seed)

#Load data from input, gotta write something for this. block

train_data_dir = '128ImagesBasic/train'
validation_data_dir = '128ImagesBasic/validation'
#nb_train_samples = 1400
#nb_validation_samples=600
epochs=100
batch_size=100				#Reduce this is we see problems. If using bluebear, might be smart to increase this. At home, use 128 max.


img_width = 128
img_height = 128
#input_shape=						#Input shape only needed for Conv

num_pixels = img_width*img_height
num_classes = 9 				#9 different categories for the output, this is tempoary workaround



#Let's define the model (simple) 

def simple_model():
    #create model - custom
        
    model = Sequential()    
    	
        #Adding additional convolution + maxpool layers 15/1/19
    model.add(Conv2D(32, (3,3), input_shape=(img_width,img_height,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(128, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
     
    model.add(Conv2D(256, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
   
   # model.add(Conv2D(512, (3,3)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.2))
    


    model.add(Flatten())		
		#Possible dense layer with our 128x128 number of pixels is too much, too high. We should add a few convolutional and maxpool layers beforehand.
    
    model.add(Dense(128,			#dimensionality of output space
			#input_shape=(128,128,1),		#Commented out as only the first layer needs input shape. 
		))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='RMSProp',metrics=['accuracy'])
    return model


def vgg_model():
    vgg16_model = applications.vgg16.VGG16(weights='imagenet',include_top=False, input_shape=(128,128,3))
    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)
    model.layers.pop()
    #for layer in model.layers:          #This freezes the layers before final layer for Transfer Learning
        #layer.trainable = False\
    model.add(Flatten())
    model.add(Dense(9, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='RMSProp',metrics=['accuracy'])
    #Doesn't work

    return model



#After data is inputed, we should augment the data in some way.
train_datagen = ImageDataGenerator(
	rescale=1./255,					#Normalized inputs from 0-255 to 0-1		
	horizontal_flip=True,
	vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#Images should be inputed the same way as the code found in main to avoid confusion
train_generator = train_datagen.flow_from_directory(
	train_data_dir,
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

callbacks = [EarlyStopping(monitor='val_loss', patience = 8),
            ModelCheckpoint(filepath='best_model.h5', monitor='acc', save_best_only=True)]


#build the model
model = simple_model()

history = model.fit_generator(
	train_generator,
	epochs=epochs,
        callbacks=callbacks,
	validation_data=validation_generator)

acc_history = history.history['acc']
val_acc_history = history.history['val_acc']

numpy_acc_history = numpy.array(acc_history)
numpy_val_acc_history = numpy.array(val_acc_history)
numpy.savetxt('acc_history.txt',numpy_acc_history, delimiter=',')
numpy.savetxt('val_acc_history.txt',numpy_val_acc_history, delimiter=',')

model.save_weights('very_simple.h5')
K.clear_session()





			

















   

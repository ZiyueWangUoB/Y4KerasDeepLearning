'''
Adapted by Ziyue Wang for Y4 project. Created on 14/1/2019
'''

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications 

#Writing a seed for reproducibility
seed = 1
numpy.random.seed(seed)

#Load data from input, gotta write something for this. block

#train_data_dir = 'unbiasedShear/shear5/A'
#train_data_dir = '128Imagesbkg515A/train'
#validation_data_dir = '128ImagesNoisyA/validation'

train_data_dir = 'unbiasedShear/shear2Point5/train'
validation_data_dir = 'unbiasedShear/shear2Point5/validation'

#nb_train_samples = 1400
#nb_validation_samples=600
epochs=200
batch_size=64				#Reduce this is we see problems. If using bluebear, might be smart to increase this. At home, use 128 max.


img_width = 128
img_height = 128
#input_shape=						#Input shape only needed for Conv

num_pixels = img_width*img_height
num_classes = 9 				#9 different categories for the output, this is tempoary workaround



#Let's define the model (simple) 

def simple_model():
    #create model - custom
        
    model = Sequential()    
    
    #model.add(Conv2D(16, (3,3), input_shape=(img_width,img_height,1)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.4))
	
        #Adding additional convolution + maxpool layers 15/1/19
    model.add(Conv2D(32, (5,5), input_shape=(img_width,img_height,1),strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
   
    
    model.add(Conv2D(128, (3,3)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
     
    '''        
    model.add(Conv2D(256, (3,3)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    
    
    model.add(Conv2D(512, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    '''


    model.add(Flatten())		
		#Possible dense layer with our 128x128 number of pixels is too much, too high. We should add a few convolutional and maxpool layers beforehand.
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(128,			#dimensionality of output space
			#input_shape=(128,128,1),		#Commented out as only the first layer needs input shape. 
		))
    model.add(Activation('relu'))
    #model.add(Dropout(0.1))
    

    #model.add(Dense(num_classes, activation='softmax'))
    model.add(Dense(num_classes,activation='softmax'))
    optimizer = RMSprop(lr=0.001) 
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy']) 
    return model



#After data is inputed, we should augment the data in some way.
train_datagen = ImageDataGenerator(
	rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

#Images should be inputed the same way as the code found in main to avoid confusion
train_generator = train_datagen.flow_from_directory(
	train_data_dir,
        color_mode='grayscale',	
	target_size=(img_width, img_height),
        batch_size=batch_size,
	class_mode='categorical',
	shuffle=True)
        #subset='training')

validation_generator = validation_datagen.flow_from_directory(
	validation_data_dir,  
        color_mode='grayscale',
        target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='categorical',
	shuffle=True)
        #subset='validation')

callbacks = [EarlyStopping(monitor='val_loss', patience = 20),
            ModelCheckpoint(filepath='best_model.h5', monitor='acc', save_best_only=True)]


#build the model
#model = simple_model()
model = simple_model()
model.summary()

history = model.fit_generator(
	train_generator,
	epochs=epochs,
        callbacks=callbacks,
	validation_data=validation_generator)

acc_history = history.history['acc']
val_acc_history = history.history['val_acc']
loss_history = history.history['loss']
val_loss_history = history.history['val_loss']


numpy_acc_history = numpy.array(acc_history)
numpy_val_acc_history = numpy.array(val_acc_history)
numpy_loss_history = numpy.array(loss_history)
numpy_val_loss_history = numpy.array(val_loss_history)

numpy.savetxt('acc_history.txt',numpy_acc_history, delimiter=',')
numpy.savetxt('val_acc_history.txt',numpy_val_acc_history, delimiter=',')
numpy.savetxt('loss_history.txt',numpy_loss_history, delimiter=',')
numpy.savetxt('val_loss_history.txt',numpy_val_loss_history, delimiter=',')



model.save_weights('very_simple.h5')
K.clear_session()





			

















   

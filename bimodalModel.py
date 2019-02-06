'''
    Bimodal model, adapted from simpleModel.py for Y4 project. Created on 5/2/2019
    '''

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ConvLSTM2D, MaxPooling2D, TimeDistributed
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications

#Writing a seed for reproducibility
seed = 1
numpy.random.seed(seed)

#Load data from input, gotta write something for this. block

train_data_dirA = '128ImagesBasicA/train'
train_data_dirB = '128ImagesBasicB/train'
validation_data_dirA = '128ImagesBasicA/validation'
validation_data_dirB = '128ImagesBasicB/validation'
epochs=100
batch_size=100                #Reduce this is we see problems. If using bluebear, might be smart to increase this. At home, use 128 max.


img_width = 128
img_height = 128
#input_shape=                        #Input shape only needed for Conv

num_pixels = img_width*img_height
num_classes = 9                 #9 different categories for the output, this is tempoary workaround



#Let's define the model (simple)

def LSTM_model():
    #create model - custom
    
    model = Sequential()
    
    #Adding additional convolution + maxpool layers 15/1/19
    model.add(ConvLSTM2D(32, (5,5), input_shape=(img_width,img_height,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(ConvLSTM2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(ConvLSTM2D(128, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(ConvLSTM2D(256, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    #Possible dense layer with our 128x128 number of pixels is too much, too high. We should add a few convolutional and maxpool layers beforehand.
    
    model.add(Dense(128,            #dimensionality of output space
                    #input_shape=(128,128,1),        #Commented out as only the first layer needs input shape.
                    ))
    model.add(Activation('relu'))
                    #model.add(Dropout(0.2))
                    
                    
                    #model.add(Dense(num_classes, activation='softmax'))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='RMSProp',metrics=['accuracy'])
    return model

def multipleInput_model():







#After data is inputed, we should augment the data in some way.
train_datagen = ImageDataGenerator(
                                   rescale=1./255,                    #Normalized inputs from 0-255 to 0-1
                                   horizontal_flip=True,
                                   vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

def generate_generator_multiple(generator,dir1, dir2, batch_size, img_width,img_height,subset):
    genX1 = generator.flow_from_directory(dir1,
                                          color_mode='grayscale',
                                          target_size=(img_width,img_height),
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          shuffle=False,
                                          subset=subset,
                                          seed=1)
                                          #Same seed for consistency.

    genX2 = generator.flow_from_directory(dir2,
                                          color_mode='grayscale',
                                          target_size=(img_width,img_height),
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          shuffle=False,
                                          subset=subset,
                                          seed=1)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0],X2i[0]],x2i[1]    #Yields both images and their mutual label


train_generator = generate_generator_multiple(generator=train_datagen,
                                              dir1=train_data_dirA,
                                              dir2=train_data_dirB,
                                              batch_size=batch_size,
                                              img_width=img_width,
                                              img_height=img_height,
                                              subset='training')

validation_generator = generate_generator_multiple(generator=test_datagen,
                                                   dir1=train_data_dirA,
                                                   dir2=train_data_dirB,
                                                   batch_size=batch_size,
                                                   img_width=img_width,
                                                   img_height=img_height,
                                                   subset='validation')

callbacks = [EarlyStopping(monitor='val_loss', patience = 8),
             ModelCheckpoint(filepath='best_model.h5', monitor='acc', save_best_only=True)]


#build the model
#model = simple_model()
model = simple_model()

history = model.fit_generator(train_generator,
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




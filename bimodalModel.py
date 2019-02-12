'''
    Bimodal model, adapted from simpleModel.py for Y4 project. Created on 5/2/2019
    '''

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ConvLSTM2D, AveragePooling2D, TimeDistributed
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications
from keras.layers.merge import concatenate


#Writing a seed for reproducibility
seed = 1
numpy.random.seed(seed)

#Load data from input, gotta write something for this. block

train_data_dirA = '128Imagesbkg10A/train'
train_data_dirB = '128Imagesbkg10B/train'
validate_data_dirA = '128Imagesbkg10A/validation'
validate_data_dirB = '128Imagesbkg10B/validation'

epochs=20
batch_size=32 #Reduce this is we see problems. If using bluebear, might be smart to increase this. At home, use 128 max.


img_width = 128
img_height = 128
#input_shape=                        #Input shape only needed for Conv

num_pixels = img_width*img_height
num_classes = 9                 #9 different categories for the output, this is tempoary workaround



#Let's define the model (simple)

def LSTM_model():
    #create model - custom
    
    model = Sequential()
    no_of_img = 2
    #Adding additional convolution + maxpool layers 15/1/19
    model.add(ConvLSTM2D(32, (5,5), input_shape=(no_of_img,img_width,img_height,1),return_sequences=True,dropout=0.2))
    model.add(Activation('relu'))
    model.add(TimeDistributed(AveragePooling2D((2,2))))
 
    model.add(ConvLSTM2D(64, (3,3),return_sequences=True,dropout=0.1))
    model.add(Activation('relu'))
    model.add(TimeDistributed(AveragePooling2D(pool_size=(2,2))))

    model.add(ConvLSTM2D(128, (3,3),return_sequences=True,dropout=0.1))
    model.add(Activation('relu'))
    model.add(TimeDistributed(AveragePooling2D(pool_size=(2,2))))
    
    '''
    #model.add(ConvLSTM2D(256, (3,3),return_sequences=True))
    #model.add(Activation('relu'))
    #model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
    #model.add(Dropout(0.2))
    '''

    model.add(Flatten())
    #Possible dense layer with our 128x128 number of pixels is too much, too high. We should add a few convolutional and maxpool layers beforehand.
    
    model.add(Dense(64,            #dimensionality of output space
                    #input_shape=(128,128,1),        #Commented out as only the first layer needs input shape.
                    ))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
                    
                    
                    #model.add(Dense(num_classes, activation='softmax'))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
    return model



#After data is inputed, we should augment the data in some way.
train_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

def generate_generator_multiple(generator,dir1, dir2, batch_size, img_width,img_height):
    genX1 = generator.flow_from_directory(dir1,
                                          color_mode='grayscale',
                                          target_size=(img_width,img_height),
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          shuffle=True,
                                          seed=1)
                                          #Same seed for consistency.

    genX2 = generator.flow_from_directory(dir2,
                                          color_mode='grayscale',
                                          target_size=(img_width,img_height),
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          shuffle=True,
                                          seed=1)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        s = numpy.stack((X1i[0],X2i[0]))
        b = numpy.transpose(s,(1,0,2,3,4))
        yield b,X1i[1]    #Yields both images and their mutual label



train_generator = generate_generator_multiple(generator=train_datagen,
                                              dir1=train_data_dirA,
                                              dir2=train_data_dirB,
                                              batch_size=batch_size,
                                              img_width=img_width,
                                              img_height=img_height)

validation_generator = generate_generator_multiple(generator=test_datagen,
                                                   dir1=validate_data_dirA,
                                                   dir2=validate_data_dirB,
                                                   batch_size=batch_size,
                                                   img_width=img_width,
                                                   img_height=img_height)




callbacks = [EarlyStopping(monitor='val_loss', patience = 8),
             ModelCheckpoint(filepath='best_model.h5', monitor='acc', save_best_only=True)]


#build the model
#model = simple_model()
model = LSTM_model()
model.summary()

history = model.fit_generator(train_generator,
                              epochs=epochs,
                              steps_per_epoch=18900 // batch_size,
                              callbacks=callbacks,
                              validation_data=validation_generator,
                              validation_steps=8100 // batch_size)

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




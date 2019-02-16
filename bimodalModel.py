'''
    Bimodal model, adapted from simpleModel.py for Y4 project. Created on 5/2/2019
    '''

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, AveragePooling2D, Conv3D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications
#from tensorflow.keras.layers.merge import concatenate
    


#Writing a seed for reproducibility
seed = 1
numpy.random.seed(seed)

#Load data from input, gotta write something for this. block

train_data_dirA = 'noisy/A'
train_data_dirB = 'noisy/B'
train_data_dirC = 'noisy/C'

epochs=100
batch_size=64 #Reduce this is we see problems. If using bluebear, might be smart to increase this. At home, use 128 max.


img_width = 128
img_height = 128
#input_shape=                        #Input shape only needed for Conv

num_pixels = img_width*img_height
num_classes = 9                 #9 different categories for the output, this is tempoary workaround



#Let's define the model (simple)

def LSTM_model():
    #create model - custom
    
    model = Sequential()
    no_of_img = 3
    model.add(Conv2D(32,(5,5),input_shape=(img_width,img_height,no_of_img),activation='relu'))
    model.add(Dropout(0.6))
    model.add(AveragePooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(Dropout(0.4))
    model.add(AveragePooling2D(pool_size=(2,2)))

    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(Dropout(0.4))
    model.add(AveragePooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(128,activation='relu'))

    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
    return model




#After data is inputed, we should augment the data in some way.
train_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,vertical_flip=True, validation_split=0.3)

#test_datagen = ImageDataGenerator(rescale=1./255)

def generate_generator_multiple(generator,dir1, dir2, dir3 ,batch_size, img_width,img_height,subset):
    genX1 = generator.flow_from_directory(dir1,
                                          color_mode='grayscale',
                                          target_size=(img_width,img_height),
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          shuffle=True,
                                          seed=1,
                                          subset=subset)
                                          #Same seed for consistency.

    genX2 = generator.flow_from_directory(dir2,
                                          color_mode='grayscale',
                                          target_size=(img_width,img_height),
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          shuffle=True,
                                          seed=1,
                                          subset=subset) 
    
    genX3 = generator.flow_from_directory(dir3,
                                          color_mode='grayscale',
                                          target_size=(img_width,img_height),
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          shuffle=True,
                                          seed=1,
                                          subset=subset) 
    
     

    
    #yield genX1.filenames,genX2.filenames,genX1.classes,genX2.classes

    
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        s = numpy.concatenate((X1i[0],X2i[0],X3i[0]),axis=3)
        #s = numpy.squeeze(s)
        #b = numpy.transpose(s,(1,2,3,0,4))
        yield s,X1i[1]   #Yields both images and their mutual label

img_width = 128
img_height = 128


validation_generator = generate_generator_multiple(generator=train_datagen,
                                              dir1=train_data_dirA,
                                              dir2=train_data_dirB,
                                              dir3=train_data_dirC,
                                              batch_size=batch_size,
                                              img_width=img_width,
                                              img_height=img_height,
                                              subset='validation')

train_generator = generate_generator_multiple(generator=train_datagen,
                                                   dir1=train_data_dirA,
                                                   dir2=train_data_dirB,
                                                   dir3=train_data_dirC,
                                                   batch_size=batch_size,
                                                   img_width=img_width,
                                                   img_height=img_height,
                                                   subset='training')




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




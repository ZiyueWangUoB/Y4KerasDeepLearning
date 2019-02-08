
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ConvLSTM2D, MaxPooling2D, TimeDistributed
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications



train_data_dirA = '128ImagesBasicA'
train_data_dirB = '128ImagesBasicB'




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
        s = numpy.stack((X1i[0],X2i[0]))
        b = numpy.transpose(s,(1,0,2,3,4))
        yield b,X1i[1]    #Yields both images and their mutual label

img = generate_generator_multiple(train_datagen,train_data_dirA,train_data_dirB,100,128,128,'training')
for i in img:
    print(i[1])



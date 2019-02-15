
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ConvLSTM2D, MaxPooling2D, TimeDistributed
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt

train_data_dirA = 'noisy/A'
train_data_dirB = 'noisy/B'
train_data_dirC = 'noisy/C'

#validate_data_dirA = '128ImagesBasicA/validate'
#validate_data_dirB = '128ImagesBasicB/validate'





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
        #s = numpy.stack((X1i[0],X2i[0],X3i[0]))
        #b = numpy.transpose(s,(1,0,2,3,4))
        yield [X1i[0],X2i[0],X3i[0]],X1i[1]   #Yields both images and their mutual label
    
batch_size = 1
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


i = 0
for x in train_generator:
    i += 1
    if i == 84:
        y = numpy.squeeze(x[0][0])
        y2 = numpy.squeeze(x[0][1])
        y3 = numpy.squeeze(x[0][2])
        print(y[64,64],y2[64,64],y3[64,64])
        plt.imshow(y,cmap='gray')
        plt.figure()
        plt.imshow(y2,cmap='gray')
        plt.figure()
        plt.imshow(y3,cmap='gray')
        print(x[1])
        plt.show()
        break

'''
for i in range(1000):
    print(validation_generator[0][i],validation_generator[1][i],validation_generator[2][i],validation_generator[3][i])
'''




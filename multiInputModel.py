'''
    Bimodal model, adapted from simpleModel.py for Y4 project. Created on 5/2/2019
    '''

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, TimeDistributed, Add, Average
from keras.layers import Input, BatchNormalization
from keras.models import Model
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications
from keras.models import load_model

#Writing a seed for reproducibility
seed = 1
numpy.random.seed(seed)

#Load data from input, gotta write something for this. block

train_data_dirA = '128Imagesbkg10A/train'
train_data_dirB = '128Imagesbkg10B/train'
validate_data_dirA = '128Imagesbkg10A/validation'
validate_data_dirB = '128Imagesbkg10B/validation'


#train_data_dirA = '128ImagesBasicA2'
#train_data_dirB = '128ImagesBasicB2'


epochs=150
batch_size=64                #Reduce this is we see problems. If using bluebear, might be smart to increase this. At home, use 128 max.


img_width = 128
img_height = 128
#input_shape=                        #Input shape only needed for Conv

num_pixels = img_width*img_height
num_classes = 9                 #9 different categories for the output, this is tempoary workaround



#Let's define the model (simple)

'''
def multiInput_model():
    #create model - custom
    
    input_1 = Input(shape=(img_width,img_height,1))
    input_2 = Input(shape=(img_width,img_height,1))   

    output_1 = Conv2D(32,(5,5), activation='relu')(input_1)
    output_1 = AveragePooling2D(pool_size=(2,2))(output_1)
    output_1 = Dropout(0.2)(output_1)
    
       
    output_1 = Conv2D(64,(3,3), activation='relu')(output_1)
    output_1 = MaxPooling2D(pool_size=(2,2))(output_1)
    output_1 = Dropout(0.1)(output_1)
    

    output_1 = Conv2D(128,(3,3), activation='relu')(output_1)
    output_1 = MaxPooling2D(pool_size=(2,2))(output_1)
    output_1 = Dropout(0.1)(output_1)
   
    
    #output_1 = Conv2D(256,(3,3), activation='relu')(output_1)
    #output_1 = MaxPooling2D(pool_size=(2,2))(output_1)
    #output_1 = Dropout(0.2)(output_1)
    

    output_1 = Flatten()(output_1)
    output_1 = Dense(128,activation='relu')(output_1)
    output_1 = Dense(num_classes,activation='softmax')(output_1)
    
    
    
    output_2 = Conv2D(32,(5,5), activation='relu')(input_2)
    output_2 = AveragePooling2D(pool_size=(2,2))(output_2)
    output_2 = Dropout(0.2)(output_2)
    
    
    output_2 = Conv2D(64,(3,3), activation='relu')(output_2)
    output_2 = MaxPooling2D(pool_size=(2,2))(output_2)
    output_2 = Dropout(0.1)(output_2)
    
    output_2 = Conv2D(128,(3,3), activation='relu')(output_2)
    output_2 = MaxPooling2D(pool_size=(2,2))(output_2)
    output_2 = Dropout(0.1)(output_2)
    
    
    #output_2 = Conv2D(256,(3,3), activation='relu')(output_2)
    #output_2 = MaxPooling2D(pool_size=(2,2))(output_2)
    #output_2 = Dropout(0.2)(output_2)
    

    output_2 = Flatten()(output_2)
    output_2 = Dense(128,activation='relu')(output_2)
    output_2 = Dense(num_classes,activation='softmax')(output_2)
    



    inputs = [input_1,input_2]
    outputs = [output_1,output_2]
    
    #output = concatenate(outputs)
    output = Average()(outputs)

    #output = Conv2D(64,(3,3), activation='relu')(output)
    #output = AveragePooling2D(pool_size=(2,2))(output)
    #output = Dropout(0.1)(output)
    
    #output = Conv2D(128,(3,3), activation='relu')(output)
    #output = AveragePooling2D(pool_size=(2,2))(output)
    #output = Dropout(0.1)(output)
       
    #output = Flatten()(output)
    #output = Dense(128,activation='relu')(output)
    #output = Dense(num_classes,activation='softmax')(output)

    model = Model(inputs,[output])
    model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])

    return model

'''

def build_base():
    
    input_ = Input(shape=(img_width,img_height,1))
    
    output = Conv2D(32,(5,5), activation='relu')(input_)
    output = AveragePooling2D(pool_size=(2,2))(output)
    output = Dropout(0.2)(output)
    
    
    output = Conv2D(64,(3,3), activation='relu')(output)
    output = AveragePooling2D(pool_size=(2,2))(output)
    output = Dropout(0.1)(output)
    
    output = Conv2D(128,(3,3), activation='relu')(output)
    output = AveragePooling2D(pool_size=(2,2))(output)
    output = Dropout(0.1)(output)
      

    output = Flatten()(output)
    #output = Dense(128,activation='relu')(output)
    #output = Dense(num_classes,activation='softmax')(output)
   
    model = Model(inputs=input_, outputs=output)
    
    return input_, output, model
    
    
def transfer_model():

    input_1, output_1, model_1 = build_base()
    input_2, output_2, model_2 = build_base()

    old_model = load_model('80modelbkg10A/best_model.h5')
    old_model.layers.pop()
    old_model.layers.pop()

    model_1.set_weights(old_model.get_weights())
    model_2.set_weights(old_model.get_weights())
    
    inputs = [input_1, input_2]
    outputs = [output_1, output_2]
    
    output = concatenate(outputs)
    output = Dense(256,activation='relu')(output)
    output = Dropout(0.3)(output)

    output = Dense(num_classes,activation='softmax')(output)
    new_model = Model(inputs,outputs=output)
    
    for layer in new_model.layers[:-2]:
        layer.trainable = False

    new_model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
    return new_model



#After data is inputed, we should augment the data in some way.
train_datagen = ImageDataGenerator(rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True)

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
        yield [X1i[0],X2i[0]],X1i[1]    #Yields both images and their mutual label



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
#model = multiInput_model()
model = transfer_model()
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




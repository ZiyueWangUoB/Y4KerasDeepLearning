
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, TimeDistributed, Add, Average
from keras.layers import Input, BatchNormalization, GlobalAveragePooling2D
from keras.models import Model
from keras.layers.merge import concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import applications
from keras.models import load_model
from keras.applications import vgg16
from keras.applications.inception_v3 import InceptionV3


seed = 1
numpy.random.seed(seed)

train_data_dir = '../../unbiasedShear/shear5/A'
#train_data_dir = '128Imagesbkg515A/train'
oldModel = 'best_model.h5'
#oldModel = 'shear/unbiased5-75/best_model.h5'
#oldModel = 'shear/unbiased2Point5-79/best_model.h5'
epochs=50
batch_size=64				#Reduce this is we see problems. If using bluebear, might be smart to increase this. At home, use 128 max.


img_width = 128
img_height = 128
#input_shape=						#Input shape only needed for Conv

num_pixels = img_width*img_height
num_classes = 9 				#9 different categories for the output, this is tempoary workaround

def transfer_model():
    old_model = load_model(oldModel)
   
    model = Sequential()

    for layer in old_model.layers[:-1]:
        model.add(layer)
    
    for layer in model.layers:
        layer.trainable = False

    model.add(Dense(9,activation='softmax',name='last_dense'))
    model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
    return model

def vgg_model():    
    #vgg part, comment out if using our own model
    old_model = vgg16.VGG16(include_top=False,weights='imagenet',input_shape=(128,128,3))


    model = Sequential()

    for layer in old_model.layers:
        model.add(layer)
    
    for layer in model.layers:
        layer.trainable = False
    
    model.add(Flatten())
    #model.add(Dense(512,activation='relu',name='some_dense'))
    model.add(Dense(128,activation='relu',name='penultimate_dense'))
    model.add(Dense(9,activation='softmax',name='last_dense'))
    model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
    return model

def inception_model():    
    #vgg part, comment out if using our own model
    #old_model = InceptionV3(include_top=False,weights='imagenet',input_shape=(128,128,3))
    base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(128,128,3))
        
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    #x = AveragePooling2D()(x)
    x = Flatten()(x) 
    x = Dense(128,activation='relu')(x)
    x = Dense(9,activation='softmax',name='last_dense')(x)

    model = Model(inputs=base_model.input, outputs=x)

    model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
    return model

#After data is inputed, we should augment the data in some way.
train_datagen = ImageDataGenerator(
	rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.3)

#Images should be inputed the same way as the code found in main to avoid confusion
train_generator = train_datagen.flow_from_directory(
	train_data_dir,
        color_mode='grayscale',	
	target_size=(img_width, img_height),
        batch_size=batch_size,
	class_mode='categorical',
	shuffle=True,
        subset='training')

validation_generator = train_datagen.flow_from_directory(
	train_data_dir,  
        color_mode='grayscale',
        target_size=(img_width, img_height),
	batch_size=batch_size,
	class_mode='categorical',
	shuffle=True,
        subset='validation')

callbacks = [EarlyStopping(monitor='val_loss', patience = 5),
            ModelCheckpoint(filepath='best_model.h5', monitor='acc', save_best_only=True)]


#build the model
#model = vgg_model()
model = transfer_model()
#model = inception_model()
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




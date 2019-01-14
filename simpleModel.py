import numpy
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Acvitation
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


#Writing a seed for reproducibility
seed = 1
numpy.random.seed(seed)

'''
#Load data from input, gotta write something for this. block

train_data_dir = '128img/train'
validation_data_dir = '128img/train'
nb_train_samples = 1400
nb_validation_samples=600
epochs=50
batch_size=64				#Reduce this is we see problems. If using bluebear, might be smart to increase this. At home, use 128 max.

'''


#After data is inputed, we should augment the data in some way.
train_datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	color_mode='grayscale',			#Converting our 3 channel RGB data into 1 channel
	horizontal_flip=True,
	vertical_flip=True)

#A need to flatten the image for multi-layer perveptron models
# flatten 128*128 pixel images.
img_width, img_height = 128, 128

num_pixels = img_width*img_height





   

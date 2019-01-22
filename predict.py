from keras.models import load_model
from keras.preprocessing import image
import numpy as np



img_width, img_height = 128, 128


model = load_model('best_Model.h5')
model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

img = image.load_img('test0.jpg', target_size=(img_width, img_height))
x = image.img_to_array(img)
#x = np.expand_dms(x, axis=0)



classes = model.predict_classes(x, batch_size=1)
print(classes)




from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import glob


img_width, img_height = 128, 128

result_list = []


model = load_model('best_model.h5')
model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

u = 0
totcount = 0
for imgname in glob.glob('alldeform/*.jpg'):
    img = image.load_img(imgname, target_size=(img_width, img_height), color_mode='grayscale')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255
    result = model.predict_classes(x)
    result_list.append(result[0,0])
    if result > 0.5:
        u +=1
    totcount +=1
    print(totcount,result[0,0])

print(result_list)
print(u)
print(float(u)/totcount)



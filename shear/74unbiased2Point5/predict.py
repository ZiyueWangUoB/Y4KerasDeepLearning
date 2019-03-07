from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import glob
import sys



img_width, img_height = 128, 128

result_list = []


model = load_model('best_model.h5')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])


u = 0
actual = 0
totcount = 0
incorrect_list = []


for imgname in glob.glob(str(sys.argv[2]) + '/*.jpg'):
    img = image.load_img(imgname, target_size=(img_width, img_height), color_mode='grayscale')
    
    img_name = str(imgname)
    newName = img_name.replace(str(sys.argv[2]) + '/','')
    newName = newName.replace('.jpg','')
    #print(newName)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255
    result = model.predict_classes(x)
    result_list.append(result[0])
    num = int(sys.argv[1])
    num_m1 = 0
    num_p1 = 0
    if num == 0:
        num_m1 = 0
        num_p1 = num + 1
    elif num == 8:
        num_p1 = 8
        num_m1 = num - 1
    else:
        num_m1 = num - 1
        num_p1 = num + 1

    res_list = [num_m1,num,num_p1]
    res = [num]

    if result[0] in res_list:
        u +=1
    
    if result[0] not in res:
        print(model.predict(x),imgname)
        incorrect_list.append(int(newName))
    else:
        actual += 1

    totcount +=1
np.save('incorrect_list',np.array(incorrect_list))
print(result_list)
print(u)
print(float(u)/totcount)
print(actual)
print(float(actual)/totcount)


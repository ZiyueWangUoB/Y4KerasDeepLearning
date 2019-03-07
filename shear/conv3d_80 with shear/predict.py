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
imgname_listA = []
for imgname in glob.glob('noisy/A/'  + str(sys.argv[2]) + '/*.jpg'):
    imgname_listA.append(imgname)

imgname_listB = [i.replace('A','B') for i in imgname_listA]    
imgname_listC = [i.replace('A','C') for i in imgname_listA] 

imgname_listAll = zip(imgname_listA,imgname_listB,imgname_listC)
   
for imgname in imgname_listAll:     
    A,B,C = imgname
    imgA = image.load_img(A, target_size=(img_width, img_height), color_mode='grayscale')
    imgB = image.load_img(B, target_size=(img_width, img_height), color_mode='grayscale')
    imgC = image.load_img(C, target_size=(img_width, img_height), color_mode='grayscale')
    #img = np.stack((imgA,imgB,imgC),axis=1)
    
    xA = image.img_to_array(imgA)
    xB = image.img_to_array(imgB)
    xC = image.img_to_array(imgC)
    x = np.stack((xA,xB,xC),axis=2)
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
    else:
        actual += 1
    
    totcount +=1

print(result_list)
print('pm 1 = ' + str(u))
print(float(u)/totcount)
print('actual = ' + str(actual))
print(float(actual)/totcount)



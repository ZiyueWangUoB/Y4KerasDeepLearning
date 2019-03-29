from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import glob
import sys
from matplotlib import pyplot as plt

def classScore(confidence_array):
    index_list = [-4,-3,-2,-1,0,1,2,3,4]
    confidence_list = confidence_array.tolist()
    #print(confidence_list)
    score_list = np.multiply(index_list,confidence_list)
    score_grad = np.diff(score_list)
    #print(score_grad,np.shape(score_grad))
    score = np.sum(score_list) #find average score
    delta = score + 4
    
    max_index = np.argmax(score_list)       #Find the index of the maximum value in the list
    if max_index == 8:
        return score, round(score+4)
    #If the gradient to the GOING TO THE RIGHT is too greater than 1, then we do the delta. 
    adjusted_score = 0
    grad = score_grad[0,max_index-1]
    if grad > 0.0 and grad < 0.2:
        #Only apply delta if gradient is greater than 0 and then smaller than 0
        adjusted_score = score - float(sys.argv[3]) #Manually enter the delta for now, based on 0 deformation
        adjusted_score += 4         #Convert back to number of deformations
        adjusted_score = round(adjusted_score)     #Round it to nearest number of deformations
        print(confidence_array,adjusted_score)
    else:
        adjusted_score = round(score+4)
    #print(adjusted_score)

    return score,int(adjusted_score)



img_width, img_height = 128, 128

result_list = []


model = load_model('best_model.h5')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

p1 = 0
m1 = 0
u = 0
actual = 0
totcount = 0
incorrect_list = []

correct_actual = []
correct_second = []

incorrect_first = []
incorrect_actual = []

grad_array = []
grad_arrayBad = []
classScore_list = []
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
    num_m1 = -1
    num_p1 = -1
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

    confidence = model.predict(x)
    confidence_sorted = -np.sort(-confidence)
    #np.swapaxes(confidence_sorted,0,1)
    #print(confidence_sorted.shape)
    #delta, adjusted_result = classScore(confidence)    #this is what I think it should be after we shift the result using 0's score
    grad_list = np.diff(confidence)
    max_index = np.argmax(confidence)
    
    grad = grad_list[0,max_index-1] #Find the gradient between the max and the one prior the max
    
    adjusted_result = result[0]

    #grad_array.append(grad)

    if grad > 0.0 and grad < 0.3:       #Try that if it's at a boundary, more likely to be the one below than the one above. 
        adjusted_result -= 1


    if result[0] in res_list:
    #if adjusted_result in res_list:
        u +=1
        #classScore_list.append(delta)
        #if result[0] == num_p1:
            #print(confidence[0,num_p1],confidence[0,num])
        if result[0] == num_p1:
            p1 += 1
        elif result[0] == num_m1:
            m1 += 1
    
    if result[0] not in res:
    #if adjusted_result not in res:
        #print(confidence,imgname)
        incorrect_list.append(int(newName))
        incorrect_first.append(confidence_sorted[0,0])
        incorrect_actual.append(confidence[0,num])
        grad_arrayBad.append(grad)

    else:
        correct_actual.append(confidence_sorted[0,0])
        #print(confidence_sorted[0,1])
        correct_second.append(confidence_sorted[0,1])
        actual += 1
        grad_array.append(grad)

    totcount +=1
plt.figure()
plt.hist([grad_array,grad_arrayBad,grad_array+grad_arrayBad],bins=5)


np.save(str(sys.argv[2]) + '/incorrect_list',np.array(incorrect_list))
#print(result_list)
plt.figure()
plt.hist(result_list,bins=5)


print(u)
print(float(u)/totcount)
print(float(p1)/totcount)
print(float(m1)/totcount)
print(actual)
print(float(actual)/totcount)
np.save(str(sys.argv[2]) + '/confidence.npy',(correct_actual,correct_second,incorrect_first,incorrect_actual))
plt.show()

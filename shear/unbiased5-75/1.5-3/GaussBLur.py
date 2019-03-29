from scipy.ndimage.measurements import center_of_mass as com
from PIL import Image
import numpy as np
from math import *
from matplotlib.pyplot import imsave
import glob
import scipy.ndimage.filters
from matplotlib import pyplot as plt
import cv2

def addPoissonNoise(Matrix):
    noise_mask = np.random.poisson(Matrix)
    return noise_mask+Matrix


paths = glob.glob('*/')
print(paths)

for path in paths:
    for filename in glob.glob(path + '*.jpg'):

        img = cv2.imread(filename)
        image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #np.asmatrix(img_array)
        #plt.imshow(image)
        
        #Average over channel dimension and squeeze
        #np.mean(image,axis=2)
        #np.asmatrix(image)
        
        #print(np.shape(image))


        gauss_blur = np.random.uniform(1.5,3)
        image = scipy.ndimage.filters.gaussian_filter(image,sigma=gauss_blur)
        image = addPoissonNoise(image)
        #print(np.shape(image))
        #np.asmatrix(image)
        #plt.imshow(image)
        #plt.show()
        
        plt.imsave(filename,image,format='jpeg',cmap='gray')
        #break
    print(path)


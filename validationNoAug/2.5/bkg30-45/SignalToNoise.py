from scipy.ndimage.measurements import center_of_mass as com
from PIL import Image
import numpy as np
from math import *
from matplotlib.pyplot import imsave
import glob
import scipy.ndimage.filters
from matplotlib import pyplot as plt
import cv2

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def addPoissonNoise(Matrix):
    noise_mask = np.random.poisson(Matrix)
    return noise_mask+Matrix


paths = glob.glob('*/')
print(paths)
SNR = []


for path in paths:
    for filename in glob.glob(path + '*.jpg'):

        img = cv2.imread(filename)
        image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        signal = np.mean(image[59:69,59:69])
        noise = np.mean(image[0:10,0:10])
        
        #snr_image = signaltonoise(image)
        #snr_image_mean = np.mean(snr_image)
        snr_image_mean = signal/noise

        SNR.append(snr_image_mean)
        #print(snr_image_mean)
        break

    print(path)


print(np.mean(SNR))

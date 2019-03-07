from PIL import Image
import numpy as np
from math import *
from matplotlib.pyplot import imsave
import glob
from skimage import io, img_as_float
from scipy import ndimage



paths = glob.glob('*/')
print(paths)

for path in paths:
    for filename in glob.glob(path+'*.jpg'):
        img = io.imread(filename,as_gray = True)
        img = img_as_float(img)
        low = np.percentile(img,10)
        img = np.subtract(img,low)


        sx = ndimage.sobel(img,axis=0,mode='constant')
        sy = ndimage.sobel(img,axis=1,mode='constant')
        sobel=np.hypot(sx,sy)

        imsave('edge/' + filename, sobel, format='jpeg',cmap='gray')
    print(path)

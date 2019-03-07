from scipy.ndimage.measurements import center_of_mass as com
from PIL import Image
import numpy as np
from math import *
from matplotlib.pyplot import imsave
import glob


paths = glob.glob('*/')
print(paths)

for path in paths:
    for filename in glob.glob(path + '*.jpg'):

        img = Image.open(filename)
        img_array = np.array(img)

        x_cen, y_cen, z = com(img_array)
        dx_cen = floor(64 - x_cen)
        dy_cen = floor(64 - y_cen)

        img_out = np.roll(img_array, dx_cen, axis=0)
        img_out = np.roll(img_out, dy_cen, axis=1)

        imsave(filename,img_out,format='jpeg',cmap='gray')
    print(path)


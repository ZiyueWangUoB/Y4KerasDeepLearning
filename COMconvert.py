from scipy.ndimage.measurements import center_of_mass as com
from PIL import Image
import numpy as np
from math import *
from matplotlib.pyplot import imsave
import glob



prePath = ['A','B','C']


paths = glob.glob('A/*/')
print(paths)

for path in paths:
    for filename in glob.glob(prePath[0] + path + '*.jpg'):
		img_A = filename
		img_B = filename.replace('A','B')	
		img_C = filename.replace('A','C')


        imgA = Image.open(img_A)
        imgA_array = np.array(imgA)

        imgB = Image.open(img_B)
        imgB_array = np.array(imgB)

        imgC = Image.open(img_C)
        imgC_array = np.array(imgC)


        Ax_cen, Ay_cen, Az = com(imgA_array)
		Bx_cen, By_cen, Bz = com(imgB_array)
		Cx_cen, Cy_cen, Cz = com(imgC_array)
		

        Bdx_cen = floor(Ax_cen - Bx_cen)
        Bdy_cen = floor(Ay_cen - By_cen)

		Cdx_cen = floor(Ax_cen - Cx_cen)
        Cdy_cen = floor(Ay_cen - Cy_cen)


        imgB_out = np.roll(img_array, Bdx_cen, axis=0)
        imgB_out = np.roll(img_out, Bdy_cen, axis=1)

		imgC_out = np.roll(img_array, Cdx_cen, axis=0)
        imgC_out = np.roll(img_out, Cdy_cen, axis=1)


        imsave(img_B,img_out,format='jpeg',cmap='gray')
		imsave(img_C,imgC_out,format='jpeg',cmap='gray')


		#Basically we leave images in A untouched, uncentered. B and C images will center to A. 


    print(path)


#Mock path - A/B/C -> ndeform -> img
		















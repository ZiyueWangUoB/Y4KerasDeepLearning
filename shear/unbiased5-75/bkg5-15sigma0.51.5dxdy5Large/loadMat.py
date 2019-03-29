from pyquaternion import Quaternion
import numpy as np
import math
from matplotlib import pyplot as plt
import glob
from math import *
import sys

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotAxisAngle(rot_mat):

        x_axis = np.array([1,0,0])
        y_axis = np.array([0,1,0])
        z_axis = np.array([0,0,1])

        axis_list = [x_axis,y_axis,z_axis]


        beam_axis = np.array([0,0,-1])

        #We rotate the three axis by the rotation matrix to see where they end up, and find the
        last_angle = 90
        axis_id = 0
        i = 0

        for axis in axis_list:
                result_axis = np.dot(rot_mat,axis)
                min_angle = angle_between(result_axis,beam_axis)
                
                if min_angle > math.pi/2:
                        min_angle = min_angle - math.pi/2

                if  min_angle < last_angle:
                        last_angle = min_angle
                        axis_id = i
                i += 1
        
        return last_angle*180/math.pi,axis_id





#Read list of all the the wrongly predicted images
img_list = np.load(str(sys.argv[1])  + '/incorrect_list.npy')
#img_list is given as the np.array of img
print(img_list)

#Arrays for all the plots

sigma = []
bkg = []
angles = []
dx = []
dy = []

allSigma = []
allAngles = []

sigmaGood = []
anglesGood = []
dxGood = []
dyGood = []

#Read txt of all the wrongly classified images
for filename in glob.glob(str(sys.argv[1]) + '/stats/*.npy'):
#for filename in glob.glob('stats/*.npy'):

    name = str(filename)            
    newName = name.replace(str(sys.argv[1]) + "/stats/","")
    newName = newName.replace('rotMat.npy','')
    
    stats = np.load(filename)
    rot_mat = stats[0]                              #we shall save all the stats in an np.array now
    angle, axis_id = rotAxisAngle(rot_mat)  
            
    allSigma.append(stats[2])
    allAngles.append(angle)

    if int(newName) in img_list:
            #stats = np.load(filename)
            #rot_mat = stats[0]                              #we shall save all the stats in an np.array now
            #angle, axis_id = rotAxisAngle(rot_mat)  
                
            angles.append(angle)
            bkg.append(stats[1])
            sigma.append(stats[2])
            dx.append(stats[3])
            dy.append(stats[4])
    else:
            sigmaGood.append(stats[2])
            anglesGood.append(angle)
            dxGood.append(stats[3])
            dyGood.append(stats[4])


#Avg confidence level?




#Plot all sigma to get idea - do we even need this?
plt.figure()
plt.hist([sigma, sigmaGood,sigma+sigmaGood],bins=10)
plt.legend(['Bad','Good','combo'])
plt.title('sigma')

plt.figure()
plt.hist([angles, anglesGood, angles+anglesGood],bins=10)
plt.legend(['Bad','Good','combo'])
plt.title('angles')


#Find hypot of dxdy 
hyp = np.hypot(dx,dy)
hypGood = np.hypot(dxGood,dyGood)


#plt.figure()
#plt.hist(sigma,bins=10)
#plt.title('sigma')

plt.figure()
plt.hist(bkg,bins=10)
plt.title('bkg')

#plt.figure()
#plt.hist(angles,bins=10)
#plt.title('angles')

#plt.figure()
#plt.hist(dx,bins=10)
#plt.title('dx')

#plt.figure()
#plt.hist(dy,bins=10)
#plt.title('dy')

plt.figure()
plt.hist([hyp, hypGood],bins=10)
plt.legend(['Bad','Good','combo'])
plt.title('hyp')

#plt.figure()
#plt.hist(hyp,bins=10)
#plt.title('hyp')

plt.show()






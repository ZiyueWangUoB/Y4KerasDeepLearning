import numpy as np
import math
from matplotlib import pyplot as plt
from math import *
import sys


confidence_tuple= np.load(str(sys.argv[1]) + '/confidence.npy')
print(np.shape(confidence_tuple))
c_act,c_second,inc_first,inc_act = confidence_tuple

c_diff = np.subtract(c_act,c_second)
inc_diff = np.subtract(inc_first,inc_act)

print(c_diff)

plt.figure()
plt.hist([c_diff,inc_diff],bins=10)
plt.legend(['c','inc'])
plt.title('diff')
plt.show()


print(np.shape(inc_act))
#avg confidence of correctly found identified images
print('avg confidence of correct = ' + str(np.mean(c_act)) + '  ' + str(np.mean(c_second)))

#avg confidence of incorrect found of incorrect label, followed by act label confidence
print('avg confidence of incorrect followed by correct when incorrectly labelled = ' + str(np.mean(inc_first)) + '   ' + str(np.mean(inc_act)))




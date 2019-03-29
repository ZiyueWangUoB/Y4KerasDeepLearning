import math
import numpy as np
import matplotlib.pyplot as plt

non = str('Non-shear')
Point25 = str('2Point5')
five = str('5')


#acc data
acc = np.array([line.rstrip(',') for line in open(non + '/acc_history.txt')]).astype(np.float)
val_acc = np.array([line.rstrip(',') for line in open(non + '/val_acc_history.txt')]).astype(np.float)

epoch = np.array([i for i in range(acc.shape[0])])

#plt.plot(epoch,acc)
plt.plot(epoch,val_acc)

i = np.argmax(val_acc)
print('vgg16noshear',val_acc[i],epoch[i])


acc = np.array([line.rstrip(',') for line in open(Point25 + '/acc_history.txt')]).astype(np.float)
val_acc = np.array([line.rstrip(',') for line in open(Point25 + '/val_acc_history.txt')]).astype(np.float)

epoch = np.array([i for i in range(acc.shape[0])])

#plt.plot(epoch,acc)
plt.plot(epoch,val_acc)

i = np.argmax(val_acc)
print('vgg2.5noshear',val_acc[i],epoch[i])


acc = np.array([line.rstrip(',') for line in open(five + '/acc_history.txt')]).astype(np.float)
val_acc = np.array([line.rstrip(',') for line in open(five + '/val_acc_history.txt')]).astype(np.float)

epoch = np.array([i for i in range(acc.shape[0])])

#plt.plot(epoch,acc)
plt.plot(epoch,val_acc)

i = np.argmax(val_acc)
print('vgg5noshear',val_acc[i],epoch[i])


plt.legend(['Non-shear','2.5%','5%'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.show()
plt.savefig('val_acc_plot.eps')



plt.figure()



#val_data
loss = np.array([line.rstrip(',') for line in open(non + '/loss_history.txt')]).astype(np.float)
val_loss = np.array([line.rstrip(',') for line in open(non + '/val_loss_history.txt')]).astype(np.float)


epoch = np.array([i for i in range(loss.shape[0])])

#plt.plot(epoch,loss)
plt.plot(epoch,val_loss)


loss = np.array([line.rstrip(',') for line in open(Point25 + '/loss_history.txt')]).astype(np.float)
val_loss = np.array([line.rstrip(',') for line in open(Point25 + '/val_loss_history.txt')]).astype(np.float)


epoch = np.array([i for i in range(loss.shape[0])])

#plt.plot(epoch,loss)
plt.plot(epoch,val_loss)


loss = np.array([line.rstrip(',') for line in open(five + '/loss_history.txt')]).astype(np.float)
val_loss = np.array([line.rstrip(',') for line in open(five + '/val_loss_history.txt')]).astype(np.float)


epoch = np.array([i for i in range(loss.shape[0])])

#plt.plot(epoch,loss)
plt.plot(epoch,val_loss)

plt.legend(['Non-shear','2.5%','5%'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.xlim([1,35])
plt.ylim([0,2])
#plt.show()
plt.savefig('val_loss_plot.eps')





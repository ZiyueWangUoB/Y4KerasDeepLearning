import math
import numpy as np
import matplotlib.pyplot as plt

#acc data
acc = np.array([line.rstrip(',') for line in open('acc_history.txt')]).astype(np.float)
val_acc = np.array([line.rstrip(',') for line in open('val_acc_history.txt')]).astype(np.float)

epoch = np.array([i for i in range(acc.shape[0])])


plt.plot(epoch,acc)
plt.plot(epoch,val_acc)
plt.legend(['Training accuracy','Validation accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.show()
plt.savefig('acc_plot.eps')

#val_data
loss = np.array([line.rstrip(',') for line in open('loss_history.txt')]).astype(np.float)
val_loss = np.array([line.rstrip(',') for line in open('val_loss_history.txt')]).astype(np.float)


epoch = np.array([i for i in range(loss.shape[0])])

plt.figure()
plt.plot(epoch,loss)
plt.plot(epoch,val_loss)
plt.legend(['Training loss','Validation loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.show()
plt.savefig('loss_plot.eps')


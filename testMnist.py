from tensorflow.python.keras.datasets import mnist

(X_train,y_train), (X_test, y_test) = mnist.load_data()

print(X_train[0])



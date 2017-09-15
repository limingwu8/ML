"""
classification using deep neural network
"""
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import numpy as np
np.random.seed(1337)

(X_train,y_train),(X_test,y_test) = mnist.load_data()   # X_train:60000x28x28,y_train:60000

# data pre-processing
X_train = X_train.reshape(X_train.shape[0],-1)/255. # normalize, X_train: 60000x784
X_test = X_test.reshape(X_test.shape[0],-1)/255.     # normalize, X_test: 60000x784
y_train = np_utils.to_categorical(y_train,num_classes=10)   # change labels to one-hot format. y_train: 60000x10
y_test = np_utils.to_categorical(y_test,num_classes=10)     # change labels to one-hot format. y_train: 60000x10
# build up neural network
model = Sequential([
    Dense(input_dim=784,units=256), # units means output size
    Activation('relu'),
    Dense(units=128),
    Activation('relu'),
    Dense(units=64),
    Activation('relu'),
    Dense(units=32),
    Activation('relu'),
    Dense(units = 10),
    Activation('softmax'),
])
# set some properties of the optimizer
rmsprop = RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)

# activate network
model.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=['accuracy'])

# train dataset
print('Testing...')
model.fit(X_train,y_train,batch_size=20,epochs=2)
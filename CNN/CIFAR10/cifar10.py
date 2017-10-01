from keras.datasets import cifar10
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from keras.optimizers import Adam

(X_train, y_train),(X_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)
print("X_train shape:%s, y_train shape:%s"%(X_train.shape,y_train.shape))
print("X_test shape:%s, y_test shape:%s"%(X_test.shape,y_test.shape))

model = Sequential()

model.add(Conv2D(
    filters = 32,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    input_shape = (32,32,3)
))
model.add(Activation('relu'))

model.add(MaxPool2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='same'
))

model.add(Conv2D(
    filters = 64,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
))
model.add(Activation('relu'))

model.add(MaxPool2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='same'
))
# Dense Layer
model.add(Flatten())
model.add(Dense(units=1024))
model.add(Activation('relu'))
model.add(Dense(units=64))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))
# optimizer
adam = Adam(lr=1e-4)
# compile
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train network
print('Training...')
model.fit(X_train,y_train,batch_size=20,epochs=2)
loss,accuracy = model.evaluate(X_test,y_test)
print("\nloss:",loss)
print("accuracy:",accuracy)
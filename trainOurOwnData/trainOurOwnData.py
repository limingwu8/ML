from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image

dir1 = "F://datasets//inputData"
dir2 = "F://datasets//inputDataResized"
# image size
imgRows , imgCols = 200, 200

# resize file
# files = os.listdir("F://datasets//inputData")
# for file in files:
#     im = Image.open(dir1 + '\\' + file)
#     img = im.resize((imgRows, imgCols))
#     gray = img.convert("L")
#     gray.save(dir2 + '\\' + file, "JPEG")

files = os.listdir("F://datasets//inputDataResized")

# example of reading a picture
# im1 = np.array(Image.open(dir2 + '\\' + files[0]))
# plt.imshow(im1)
# m, n = im1.shape[0:2]
# imnbr = len(files)

# create matrix to store all flattened images
# imMatrix = []
# for i in files:
#     with Image.open(dir2 + '\\' + i) as im:
#         flatten = np.array(im).flatten()
#         imMatrix.append(flatten)
# imMatrix = np.array(imMatrix)
# convert the above to one line
imMatrix = np.array([np.array(Image.open(dir2 + '\\' + im)).flatten() for im in files]) # array(3088x40000)

label = np.ones(len(files),dtype=int)
label[:1485] = 0    # cats class
label[1485:] = 1    # dogs class

data,label = shuffle(imMatrix,label,random_state = 2)

trainData = [data,label]

# batch size to train
batchSize = 32
# number of output classes
outputClasses = 2
# number of channels
imgChannels = 1
# number of convolutional filters to use
filters = 32
# kernal size
kernalSize = 5
# max pooling size
poolSize = 2
# convolutional kernal size
convSize = 3

(X,y) = (trainData[0],trainData[1])

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)   # X_train: (2470x40000), y_train: (2470,)
# data preprocess
X_train = X_train.reshape(X_train.shape[0],1,imgRows,imgCols)   # X_train: (2470x1x200x200)
X_test = X_test.reshape(X_test.shape[0],1,imgRows,imgCols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255  # normalize
X_test = X_test/255  # normalize
y_train = np_utils.to_categorical(y_train,outputClasses)    # convert to hot key format, cat,0,[1,0]. dog, 1,[0,1]
y_test = np_utils.to_categorical(y_test,outputClasses)

# construct convolutional neural network
model = Sequential()
# Conv layer 1 output size : 32x200x200
model.add(Conv2D(
    filters = 32,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    input_shape = (1,imgRows,imgCols)
))
model.add(Activation('relu'))
model.add(Conv2D(
    filters = 32,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    input_shape = (1,imgRows,imgCols)
))
model.add(Activation('relu'))
# MaxPooling layer 1 output shape : 32x100x100
model.add(MaxPool2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='same'
))
model.add(Dropout(0.5))

# Dense Layer
model.add(Flatten())
model.add(Dense(units=128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(units=2))
model.add(Activation('softmax'))

# compile
model.compile(optimizer="adam",
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# train network
print('Training...')

model.fit(X_train,y_train,batch_size=10,epochs=20,verbose=1,validation_split=0.2)
loss,accuracy = model.evaluate(X_test,y_test)
print("\nloss:",loss)
print("accuracy:",accuracy)
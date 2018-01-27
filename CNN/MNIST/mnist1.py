import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPool2D, Activation, Dense, Flatten
from keras.optimizers import Adam
from keras.models import Sequential
from keras.optimizers import RMSprop

np.random.seed(1337)

(X_train,y_train),(X_test,y_test) = mnist.load_data()   # X_train:60000x28x28,y_train:60000

# data pre-processing
X_train = X_train.reshape(-1,1,28,28)   # X_train: 60000x1x28x28
X_test = X_test.reshape(-1,1,28,28)     # X_test: 60000x1x28x28
y_train = np_utils.to_categorical(y_train,num_classes=10)   # change labels to one-hot format. y_train: 60000x10
y_test = np_utils.to_categorical(y_test,num_classes=10)     # change labels to one-hot format. y_train: 60000x10
print("X_train shape:%s, y_train shape:%s"%(X_train.shape,y_train.shape))
print("X_test shape:%s, y_test shape:%s"%(X_test.shape,y_test.shape))
model = Sequential()

# Conv layer 1 output size : 32x28x28
model.add(Conv2D(
    filters = 32,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    input_shape = (1,28,28)
))
print(model)
model.add(Activation('relu'))
# MaxPooling layer 1 output shape : 32x14x14
model.add(MaxPool2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='same'
))
# Conv layer 2 output size : 64x14x14
model.add(Conv2D(
    filters = 64,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
))
model.add(Activation('relu'))
# MaxPooling layer 2 output shape : 64x7x7
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
# model.save('/home/liming/mnist.h5')
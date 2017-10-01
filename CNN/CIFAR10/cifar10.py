from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

(X_train, y_train),(X_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)
print("X_train shape:%s, y_train shape:%s"%(X_train.shape,y_train.shape))
print("X_test shape:%s, y_test shape:%s"%(X_test.shape,y_test.shape))

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(80, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

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
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)
model.fit(X_train,y_train,
          batch_size=32,
          epochs=200,
          callbacks=[tensorboard])
loss,accuracy = model.evaluate(X_test,y_test)
print("\nloss:",loss)
print("accuracy:",accuracy)
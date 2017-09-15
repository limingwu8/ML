# use RNN to classify handwriting numbers.
from keras.layers import SimpleRNN,Activation,Dense
from keras.utils import np_utils
from keras.models import Sequential
from keras.datasets import mnist
from keras.optimizers import Adam
import numpy as np
np.random.seed(1337)

TIME_STEPS = 28
INPUT_SIZE = 28
BATCH_INDEX = 0
BATCH_SIZE = 50
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001

(X_train,y_train),(X_test,y_test) = mnist.load_data()   # X_train:60000x28x28,y_train:60000
# the format of each picture is 28x28 pixel,in order to use RNN, firstly, serialize data, that is
# every line is a input unit, so input size = 28, there are totally 28 lines, thus the step length = 28.
X_train = X_train.reshape(-1,28,28)/255.    # normalize
X_test = X_test.reshape(-1,28,28)/255.      # normalize
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

# build RNN model
model = Sequential()

# RNN cell
model.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size
    # otherwise, model.evaluate() will get error.
    input_dim = INPUT_SIZE,
    input_shape=(TIME_STEPS,CELL_SIZE),
    units = CELL_SIZE,
    unroll = True
))
# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
for step in range(8001):
    X_batch = X_train[BATCH_INDEX : BATCH_INDEX + BATCH_SIZE]
    Y_batch = y_train[BATCH_INDEX : BATCH_INDEX + BATCH_SIZE]
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX>X_train.shape[0] else BATCH_INDEX

    if step % 500 == 0:
        cost, accuracy = model.evaluate(X_test,y_test,batch_size=y_test.shape[0],verbose=False)
        print('test cost: ',cost, 'test accuracy: ', accuracy)
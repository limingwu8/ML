import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
np.random.seed(1337)

# create some data
X = np.linspace(-1,1,200)
np.random.shuffle(X)
Y = 0.5 * X + np.random.normal(0,0.1,(200,))
# plot data
plt.scatter(X,Y)
plt.show()

X_train, Y_train = X[:160],Y[:160]  # train the first 160 data points
X_test, Y_test = X[160:],Y[160:]    # train the last 40 data sets

# build the model
model = Sequential()
model.add(Dense(output_dim=1,input_dim=1))
# activate the model
model.compile(loss = 'mse',optimizer='sgd')     # loss function(mses:均方误差) and optimizing method(sgd:随机梯度)

# train the model
print('Training...')
for step in range(301):
    cost = model.train_on_batch(X_train,Y_train)
    if step % 100 == 0:
        print('train cost:',cost)

# Test the model
print('Testing...')
cost = model.evaluate(X_test,Y_test,batch_size=40)
print('test cost:',cost)
W,b = model.layers[0].get_weights()
print('Weights=',W,' biases=',b)

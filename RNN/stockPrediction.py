# predict the stock price
# sp500.csv, the values are closing price

from keras import lstm

X_train,y_train,X_test,y_test = lstm.load_data('sp500.csv', 50, True)

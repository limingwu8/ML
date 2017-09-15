# sentdex, Machine Learning #14
# classify breast cancer type(benign or malignant) by using KNN
import numpy as np
from sklearn import model_selection, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True) # Do not count this point which has value '?'
df.drop('id',1,inplace=True)        # id is the useless column, if include id column, the accuracy is 0.56

x = np.array(df.drop('class',1))    # features
y = np.array(df['class'])   # class

# shuffle the data set to train and test
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size=0.2)
# get a KNN classifier
clf = neighbors.KNeighborsClassifier(n_neighbors = 27)
# train this classifier
clf.fit(x_train,y_train)

# test the accuracy of the KNN model
accuracy = clf.score(x_test,y_test)
print(accuracy)

# make a prediction
example_measures = np.array([4,3,1,1,1,2,3,2,1])
print(example_measures.shape)   # 9
example_measures = example_measures.reshape(1,-1)#-1 simply means that it is an unknown dimension and we want numpy to figure it out.
# example_measures = np.array([[4,3,1,1,1,2,3,2,1],[4,5,2,5,6,6,7,1,1]])
# example_measures = example_measures.reshape(len(example_measures),-1)
prediction = clf.predict(example_measures)
print(prediction) # [2 4]

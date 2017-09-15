# creating our own KNN algorithms
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_features = [4,4]
# plot the data
for i in dataset:
    for j in dataset[i]:
        plt.scatter(j[0],j[1],s=100,color=i)
# [[plt.scatter(j[0],j[1],s=100,color=i) for j in dataset[i]] for i in dataset] # the above loops equals this one line.
plt.scatter(new_features[0],new_features[1],s=100,color='gray')
plt.show()

def k_nearest_neighbors(data,predict,k=3):
    if k <= len(data):
        warnings.warn('K is set to a value less than total voting groups! idiot!')
    distances = []
    for group in data:  # group will be 'k' or 'r'
        for features in data[group]:
            # euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2)) # it's slower than the following line, but they do the same thing
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))   # calculate the norm, means the square
            distances.append([euclidean_distance,group]) # calculate all distances between data and predict data
    votes = [i[1] for i in sorted(distances)[:k]]   # ['k', 'r', 'k']
    vote_results = Counter(votes).most_common(1)[0][0]    # list, [('k', 2)]
    return vote_results

result = k_nearest_neighbors(dataset,new_features,3)
print(result)
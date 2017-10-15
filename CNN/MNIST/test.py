import numpy as np

a = []
for i in range(10):
    a.append(i)

print(a)

b = np.array([np.array(i) for i in range(10)])
print(b)
print(b.shape)
print(b[0])
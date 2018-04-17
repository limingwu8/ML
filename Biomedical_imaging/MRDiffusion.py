import scipy.io
import numpy as np

# read mat file
mat = scipy.io.loadmat('dtidata.mat')


dtidata = mat['dtidata']
overlay = np.matrix.copy(dtidata)

# Weighting direction scheme, ndarray(6,3)
G = np.array([[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, 1, -1], [1, 1, 0], [-1, 1, 0]])
b = 1000

s = np.zeros(7)
A = np.zeros((6, 6))
B = np.zeros((6, 1))
D = np.zeros((6, 128, 128, 75))
EV = np.zeros((3, 128, 128, 75))
FA = np.zeros((1, 128, 128, 75))
for i in range(dtidata.shape[1]):
    for j in range(dtidata.shape[2]):
        for k in range(dtidata.shape[3]):
            print(i, ' ', j, ' ', k)
            for q in range(dtidata.shape[0]):
                s[q] = dtidata[q][i][j][k]
            if s[0]==0:
                continue
            for q in range(dtidata.shape[0]-1): # 0-5
                B[q] = (1/(-b))*np.log(s[q+1]/s[0])
                A[q] = np.array([G[q][0]**2, 2*G[q][0]*G[q][1], 2*G[q][0]*G[q][2], G[q][1]**2, 2*G[q][1]*G[q][2], G[q][2]**2])
            if np.count_nonzero(B) == 0:
                continue
            d = np.squeeze(np.linalg.solve(A, B))
            D[:, i, j, k] = d

            ev = abs(np.linalg.eig(np.array([[d[0], d[1], d[2]], [d[1], d[3], d[4]], [d[2], d[4], d[5]]]))[0])
            ev_bar = np.sum(ev)/3
            EV[:, i, j,k] = ev


            fa = np.sqrt(3*((ev[0]-ev_bar)**2 + (ev[1]-ev_bar)**2 + (ev[2]-ev_bar)**2)/(2*(np.sum(np.power(ev,2)))))
            if np.isnan(fa):
                print('find nan')
            if fa > 0.25:
                overlay[0, i, j, :] = 0
            FA[:, i, j, k] = fa


# solve linear equation
# 3*a + b = 9
# a + 2*b = 8
A = np.array([[3, 1], [1, 2]])
B = np.array([[9], [8]])
a,b = np.linalg.solve(A, B)
print(a, ' ', b)


from __future__ import division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt

X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T


#Building Xbar
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one,X), axis=1)

#Calculate weights of the fitting line
A=np.dot(Xbar.T, Xbar)
b=np.dot(Xbar.T, y)
w=np.dot(np.linalg.pinv(A),b)
print('w=', w)

#preparing the fitting line
w_0=w[0][0]
w_1=w[1][0]
x0=np.linspace(145,185,2)
y0=w_0+w_1*x0


#Visualize data
plt.plot(X.T, y.T, 'ro')
plt.plot(x0,y0)
plt.legend()
plt.show()
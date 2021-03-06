print(__doc__)

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

y[::5] += 5* (0.5 - np.random.rand(8))

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)

y_rbf = svr_rbf.fit(X,y).predict(X)
y_lin = svr_lin.fit(X,y).predict(X)
y_poly = svr_poly.fit(X,y).predict(X)

lw = 2
plt.scatter(X, y, color='green', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color = 'red', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='yellow', lw = lw, label = 'Poly model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support vector regression')
plt.legend()
plt.show()

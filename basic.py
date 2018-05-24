import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
iris_X = iris.data
iris_y = iris.target

#print(data.shape)
#rint(iris.DESCR)

iris_y_unique = np.unique(iris_y)
#print(iris_y_unique)
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)

print(iris_X_train)
print(iris_y_train)
print('predict----------')
print(knn.predict(iris_X_test))
print(iris_y_test)

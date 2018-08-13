from __future__ import print_function
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

boston=datasets.load_boston()
X=pd.DataFrame(boston.data)
y=pd.DataFrame(boston.target)

X_train = X[:400]
y_train=y[:400]
X_test=X[401:]
y_test=y[401:]
print(X_train)
print(y_train)
lm = linear_model.LinearRegression()
#Traing
lm.fit(X_train, y_train)

#Predict
pre = lm.predict(X_test)

plt.scatter(y_test, pre)
plt.show()



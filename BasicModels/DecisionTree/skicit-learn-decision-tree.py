from __future__ import print_function
from sklearn import tree
X=[[0,0],[1,1]] #[n_samples, n_features]
Y=[0,1] #[n_samples]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
pre0=clf.predict([[0.1,0.1]])# ----> 0
print(pre0)
pre2 = clf.predict([[2., 2.]]) #----> 1
print(pre2)
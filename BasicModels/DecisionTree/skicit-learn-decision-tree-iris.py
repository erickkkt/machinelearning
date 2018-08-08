from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

iris = load_iris()
X=iris.data
y=iris.target
clf = tree.DecisionTreeClassifier()
#Training
clf=clf.fit(X,y)

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")

# graph

#Predict
pre = clf.predict(iris.data[:1, :])

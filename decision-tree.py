# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from iris import *

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105, 150))


export_graphviz(tree, out_file='dot/tree.dot', feature_names=['petal length', 'petal width'])

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

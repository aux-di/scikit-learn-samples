# -*- coding: utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier
from iris import *

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier=knn, test_idx=range(105, 150))

plt.xlabel('petal length [starndardized]')
plt.ylabel('petal width [starndardized]')
plt.legend(loc='upper left')
plt.show()

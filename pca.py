# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from iris import *

pca = PCA(n_components=2)
lr = LogisticRegression()

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.fit_transform(X_test_std)

lr.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, classifier=lr)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='upper left')
plt.show()

plot_decision_regions(X_test_pca, y_test, classifier=lr)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='upper left')
plt.show()

pca = PCA(n_components=None)

X_train_pca = pca.fit_transform(X_train_std)

print pca.explained_variance_ratio_
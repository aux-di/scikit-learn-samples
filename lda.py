# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from iris import *

lda = LDA(n_components=2)
lr = LogisticRegression()

X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier=lr)

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='upper left')
plt.show()

plot_decision_regions(X_test_lda, y_test, classifier=lr)

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='upper left')
plt.show()

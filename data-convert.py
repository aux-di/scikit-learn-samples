# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

'''
1
'''

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']

print df

size_mapping = {'XL': 3, 'L': 2, 'M': 1}

df['size'] = df['size'].map(size_mapping)

print df

class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}

print class_mapping

df['classlabel'] = df['classlabel'].map(class_mapping)

print df

inv_class_mapping = {v: k for k, v in class_mapping.items()}

print inv_class_mapping

df['classlabel'] = df['classlabel'].map(inv_class_mapping)

print df

class_le = LabelEncoder()

y = class_le.fit_transform(df['classlabel'].values)

print y

inv_y = class_le.inverse_transform(y)

print inv_y

'''
2
'''

X = df[['color', 'size', 'price']].values

print X

color_le = LabelEncoder()

X[:, 0] = color_le.fit_transform(X[:, 0])

print X

ohe = OneHotEncoder(categorical_features=[0])

X = ohe.fit_transform(X).toarray()

print X

X = pd.get_dummies(df[['price', 'color', 'size']])

'''
3
'''

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of Ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', '0D280/0D315 of diluted wines', 'Proline']

print('Class labels', np.unique(df_wine['Class label']))

print df_wine.head()

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

mms = MinMaxScaler()

X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.fit_transform(X_test)

print X_train_norm

stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

print X_train_std

'''
4
'''

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

forest.fit(X_train, y_train)

importances = forest.feature_importances_

print importances

indices = np.argsort(importances)[:: -1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

X_selected = forest.transform(X_train, threshold=0.15)

print X_selected.shape
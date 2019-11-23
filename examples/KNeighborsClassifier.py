#!/usr/bin/env python3

import timeit

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

# yamlfs implementations
import sys
sys.path.append("..")
from yamlfs.supervised import k_neighbors_classifier

# scikit implementations
from sklearn import neighbors

seed = 16

# Load iris data for example
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

X_train = pd.DataFrame(X_train, columns=['sepal_length',
                                         'sepal_width',
                                         'petal_length',
                                         'petal_width'])
X_test = pd.DataFrame(X_test, columns=['sepal_length',
                                       'sepal_width',
                                       'petal_length',
                                       'petal_width'])
y_train = pd.DataFrame(y_train, columns=['target'])
y_test = pd.DataFrame(y_test, columns=['target'])

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)


### K-Nearest Neighbors
print("**KNeighborsClassifier**")

# yamlfs
start_time = timeit.default_timer()
knn_yamlfs = k_neighbors_classifier.KNeighborsClassifier(neighbors=5)
knn_yamlfs.fit(X_train, y_train)
pred_knn_yamlfs = knn_yamlfs.predict(X_test)
elapsed = timeit.default_timer() - start_time

print('yamlfs Accuracy: ' + str(accuracy_score(y_test, pred_knn_yamlfs)))
print('Time Elapsed: ' + str(elapsed))

# scikit
start_time = timeit.default_timer()
knn_scikit = neighbors.KNeighborsClassifier(n_neighbors=5)
knn_scikit.fit(X_train, y_train.values.ravel())
pred_knn_scikit = knn_scikit.predict(X_test)
elapsed = timeit.default_timer() - start_time

print('Scikit Accuracy: ' + str(accuracy_score(y_test, pred_knn_scikit)))
print('Time Elapsed: ' + str(elapsed))

print('-------------------------------------')

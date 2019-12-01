#!/usr/bin/env python3

import timeit

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd

# npml implementations
from npml.supervised import logistic_regression

# scikit implementations
from sklearn import linear_model

seed = 16

# Load random blob dataset
X, y = datasets.make_classification(n_samples=10000, n_features=10, n_classes=3, n_informative=4)
X = pd.DataFrame(X)
y = pd.DataFrame(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Softmax Regression
print("**Softmax Regression**")

# npml
start_time = timeit.default_timer()
sr_npml = logistic_regression.SoftmaxRegression(optimization='radam', verbose=1)
sr_npml.fit(X_train, y_train)
pred_sr_npml = sr_npml.predict(X_test)
elapsed = timeit.default_timer() - start_time

print('npml Accuracy: ' + str(accuracy_score(y_test, pred_sr_npml)))
print('Time Elapsed: ' + str(elapsed))

# scikit
"""
I am using a gradient descent optimization method. SKLearn uses the liblinear solver, which has a
different and much faster optimization technique.
"""

start_time = timeit.default_timer()
lr_scikit = linear_model.LogisticRegression(verbose=1)
lr_scikit.fit(X_train, y_train.values.ravel())
pred_lr_scikit = lr_scikit.predict(X_test)
elapsed = timeit.default_timer() - start_time

print('Scikit Accuracy: ' + str(accuracy_score(y_test, pred_lr_scikit)))
print('Time Elapsed: ' + str(elapsed))

print('-------------------------------------')


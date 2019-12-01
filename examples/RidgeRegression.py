#!/usr/bin/env python3

import timeit

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# npml implementations
from npml.supervised import linear_regression

# scikit implementations
from sklearn import linear_model

seed = 16

# Load random blob dataset
num_observations = 500
data = np.random.multivariate_normal([0, 0], [[1, .9],[.9, 1]], num_observations)
X = pd.DataFrame(np.vstack(data[:, 0]))
y = pd.DataFrame(data[:, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Ridge Regression
print("**Ridge Regression**")

# npml
start_time = timeit.default_timer()
rr_npml = linear_regression.RidgeRegression(optimization='radam', verbose=1)
rr_npml.fit(X_train, y_train)
pred_rr_npml = rr_npml.predict(X_test)
elapsed = timeit.default_timer() - start_time

print('npml MSE: ' + str(mean_squared_error(y_test, pred_rr_npml)))
print('Time Elapsed: ' + str(elapsed))

# scikit
"""
I am using a gradient descent optimization method. SKLearn uses the liblinear solver, which has a
different and much faster optimization technique.
"""

start_time = timeit.default_timer()
rr_scikit = linear_model.Ridge()
rr_scikit.fit(X_train, y_train.values.ravel())
pred_rr_scikit = rr_scikit.predict(X_test)
elapsed = timeit.default_timer() - start_time

print('Scikit MSE: ' + str(mean_squared_error(y_test, pred_rr_scikit)))
print('Time Elapsed: ' + str(elapsed))

print('-------------------------------------')

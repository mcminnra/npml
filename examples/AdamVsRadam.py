#!/usr/bin/env python3

import timeit

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# npml implementations
from npml.supervised import linear_regression

# Load random blob dataset
num_observations = 500
data = np.random.multivariate_normal([0, 0], [[1, .9], [.9, 1]], num_observations)
X = pd.DataFrame(np.vstack(data[:, 0]))
y = pd.DataFrame(data[:, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normal LR with Normal Gradient Descent
start_time = timeit.default_timer()
lr_npml = linear_regression.RidgeRegression(optimization=None)
lr_npml.fit(X_train, y_train)
pred_lr_npml = lr_npml.predict(X_test)
elapsed = timeit.default_timer() - start_time

print('LR Normal Gradient Descent MSE: ' + str(mean_squared_error(y_test, pred_lr_npml)))
print('Time Elapsed: ' + str(elapsed))

# Normal LR with Adam Gradient Descent
start_time = timeit.default_timer()
lr_npml = linear_regression.RidgeRegression(optimization='adam')
lr_npml.fit(X_train, y_train)
pred_lr_npml = lr_npml.predict(X_test)
elapsed = timeit.default_timer() - start_time

print('LR Adam MSE: ' + str(mean_squared_error(y_test, pred_lr_npml)))
print('Time Elapsed: ' + str(elapsed))

# Normal LR with Rectified Adam Gradient Descent
start_time = timeit.default_timer()
lr_npml = linear_regression.RidgeRegression(optimization='radam')
lr_npml.fit(X_train, y_train)
pred_lr_npml = lr_npml.predict(X_test)
elapsed = timeit.default_timer() - start_time

print('LR RAdam MSE: ' + str(mean_squared_error(y_test, pred_lr_npml)))
print('Time Elapsed: ' + str(elapsed))

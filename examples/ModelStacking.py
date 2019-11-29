#!/usr/bin/env python

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from npml.util.stacking import stack

# Generate a sample dataset
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=7, n_redundant=1, n_classes=2)

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# List of Models
models = [
    LogisticRegression(solver='lbfgs'),
    RandomForestClassifier(n_estimators=100)
]

# Get Stacked X
X_train_l2, X_test_l2 = stack(models, X_train, y_train, X_test)

# Individual Accuracy
print('--Individual Accuracy--')
for model in models:
    accuracy = model.fit(X_train, y_train).score(X_test, y_test)
    print(f'{model.__class__.__name__} Accuracy: {accuracy:0.4f}')

# Stacked Accuracy
print('\n--Stacked Accuracy--')
stacked_model = LogisticRegression(solver='lbfgs')
stacked_accuracy = stacked_model.fit(X_train_l2, y_train).score(X_test_l2, y_test)
print(f'Stacked Accuracy: {stacked_accuracy:0.4f}')

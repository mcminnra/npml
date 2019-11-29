# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def get_out_of_fold(model, X_train, y_train, X_test, k=5, random_state=42):
    """
    Gets Out-Of-Fold Predictions for a model

    Parameters
    ----------
    model : list of pandas.DataFrame
        Model to get Out-Of-Fold predictions for
    X_train : numpy.ndarray or pandas.DataFrame
        Features of Training Set
    y_train : numpy.ndarray or pandas.DataFrame
        Target of Training Set
    X_test : numpy.ndarray or pandas.DataFrame
        Featrues of Testing Set
    k : int
        Number of folds
    random_state : int
        The seed of the pseudo random number generator to use when shuffling the data.

    Returns
    -------
    numpy.array
        Out-Of-Fold X Train
    numpy.array
        Out-Of-Fold X Test

    Raises
    ------
    TypeError
        when X_train, y_train, or X_test isn't a Numpy ndarray or a Pandas DataFrame
    """
    # Check Input
    if (not isinstance(X_train, (np.ndarray, pd.Series, pd.DataFrame))
            or not isinstance(y_train, (np.ndarray, pd.Series, pd.DataFrame))
            or not isinstance(X_test, (np.ndarray, pd.Series, pd.DataFrame))):
        raise TypeError(
            'Input Data must be either a Numpy ndarray, Pandas Series, or Pandas DataFrame')

    # Convert to Numpy Array
    if isinstance(X_train, (pd.Series, pd.DataFrame)):
        X_train = X_train.values
    if isinstance(y_train, (pd.Series, pd.DataFrame)):
        y_train = y_train.values
    if isinstance(X_test, (pd.Series, pd.DataFrame)):
        X_test = X_test.values

    # Create Folds
    kf = KFold(n_splits=k, random_state=random_state)

    # Init oof predictions arrays
    oof_train = np.zeros((len(X_train),))
    oof_test = np.zeros((len(X_test),))

    # Create matrix to hold X_test predictions across folds
    # The oof esimations you apply to train, you need to apply to X_test as well.
    oof_test_folds = np.empty((k, len(X_test)))

    # For each fold, create predictions on the out fold (out-of-fold predictions)
    for i, (train_idx, test_idx) in enumerate(kf.split(X_train, y_train)):
        # Train on "in-folds"
        X_in_folds = X_train[train_idx]
        y_in_folds = y_train[train_idx]

        # Predict "Out Folds"
        X_out_fold = X_train[test_idx]

        # Fit Model
        model.fit(X_in_folds, y_in_folds)

        # Make Out-Of-Fold Predictions
        oof_train[test_idx] = model.predict(X_out_fold)
        oof_test_folds[i, :] = model.predict(X_test)

    # Take the mean of test across all folds
    oof_test[:] = oof_test_folds.mean(axis=0)

    # Return new X_train, and X_test as numpy arrays for the model.
    return oof_train, oof_test


def stack(models, X_train, y_train, X_test, k=5, random_state=42):
    """
    Gets Out-Of-Fold predictions for a list of models

    Parameters
    ----------
    models : list of scikit api models
        List of models to stack
    X_train : pandas.DataFrame
        Features of Training Set
    y_train : pandas.DataFrame
        Target of Training Set
    X_test : pandas.DataFrame
        Features of Testing Set
    k : int
        Number of folds
    random_state : int
        The seed of the pseudo random number generator to use when shuffling the data.

    Returns
    -------
    pandas.DataFrame
        DataFrame of the predictions of each of the models
    """
    # Init stacked dicts
    X_train_stacked = {}
    X_test_stacked = {}

    # for each model, create out of fold predictions
    for i, model in enumerate(models):
        X_train_oof, X_test_oof = get_out_of_fold(
            model, X_train, y_train, X_test, k=k, random_state=random_state)

        X_train_stacked[f'model_{i+1}'] = X_train_oof
        X_test_stacked[f'model_{i+1}'] = X_test_oof

    return pd.DataFrame(X_train_stacked), pd.DataFrame(X_test_stacked)

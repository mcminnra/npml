# -*- coding: utf-8 -*-

import pandas as pd
from scipy import stats


def null_check(df):
    """
    Prints a sorted list of columns that contain null values and the percentage
    of the column that is null.

    Parameters
    ----------
    df : Pandas DataFrame
        Input dataframe

    Returns
    -------
    Pandas DataFrame
        A dataframe containing the count and percentages of missing values in
        the input dataframe

    Raises
    ------
    TypeError
        when a the input isn't a Panda DataFrame
    """
    if type(df) != pd.core.frame.DataFrame:
        raise TypeError("Input must be a Pandas DataFrame.")

    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() /
               df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent],
                             axis=1,
                             keys=['Total', 'Percent'])

    return missing_data


def type_check(df):
    """
    Returns a dictionary of dataframe columns grouped by their types

    Parameters
    ----------
    df : Pandas DataFrame

    Returns
    -------
    dict
        A dictionary of dataframe columns grouped by their types

    Raises
    ------
    TypeError
        when a the input isn't a Panda DataFrame
    """
    if type(df) != pd.core.frame.DataFrame:
        raise TypeError("Input must be a Pandas DataFrame.")

    g = df.columns.to_series().groupby(df.dtypes).groups
    return {k.name: v for k, v in g.items()}


def is_normal(X, alpha=0.05, verbose=False):
    """
    Checks array for normality

    Parameters
    ----------
    X : array_like
        Array of numbers to run normality test
    alpha : float, optional
        Alpha parameter for p-test
    verbose : bool, optional
        Verbosity flag

    Returns
    -------
    bool
        Boolean on whether X passed normal test

    """
    # FIXME: Need test for this
    _, p = stats.normaltest(X)

    print("p = {:g}".format(p))

    if p < alpha:  # null hypothesis: x comes from a normal distribution
        if verbose:
            print('The null hypothesis can be rejected.')
            print('Not a Normal distribution.')
        return False
    else:
        if verbose:
            print('The null hypothesis cannot be rejected.')
            print('May be a Normal Distribution.')
        return True

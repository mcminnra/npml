# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats

def feature_selection_correlation(X, y, k, return_drop_list=False, verbose=False):
    """
    Calculates correlation between features and target, then returns top k features

    Parameters
    ----------
    X : Pandas DataFrame
        Feature set
    y : Pandas Series
        Target variable
    k : int
        Number of most correlated features to take
    return_drop_list : bool, optional
        If set to true, will return a list of columns to drop instead of a full DataFrame 
    verbose : bool, optional
        Verbosity flag

    Returns
    -------
    Pandas DataFrame or list of X columns
        Top k correlated features from X
    """
    if k > len(X.columns):
        raise IndexError('k larger than number of X features')

    # Get absolute correlation by feature
    corr_dict = {}
    columns = X.columns.values
    for col in columns:
        corrcoef, _ = stats.pearsonr(X[col], y)
        corr_dict[col] = np.absolute(corrcoef)

    # Sort into a tuple list of most correlated
    sorted_corr_list = sorted(corr_dict.items(), key=lambda x: x[1], reverse=True)

    # print tuple list
    if verbose:
        print('== Features Sorted by Absolute Correlation ==')
        for i, (col, abs_corr) in enumerate(sorted_corr_list):
            print(f'{i+1}: {col} - {abs_corr}')

    # Get features to drop from list
    sorted_feature_list = [k for k, v in sorted_corr_list]
    drop_list = sorted_feature_list[k:]

    if verbose:
        print(f'k = {k} -- Dropping: {drop_list}')

    if return_drop_list:
        return drop_list
   
    X = X.drop(drop_list, axis=1)
    return X

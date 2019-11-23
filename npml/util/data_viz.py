# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def corr_matrix_plot(df, figsize=(14, 10)):
    """
    Creates a correlation matrix from the dataframe and return the matplotlib
    figure

    Parameters
    ----------
    df : Pandas DataFrame
        Input DataFrame
    figsize : tuple of ints, optional
        Size of the correlation matrix

    Returns
    -------
    fig, ax
        Matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    corr = df.corr()
    sns.heatmap(corr,
                mask=np.zeros_like(corr, dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True,
                ax=ax,
                annot=True)
    return fig, ax

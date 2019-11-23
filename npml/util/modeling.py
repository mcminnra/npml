# -*- coding: utf-8 -*-

import timeit

import numpy as np
from sklearn.model_selection import cross_val_score
from termcolor import colored


def estimator_report(output_run_estimators, digits=6, print_params=False):
    """Prints a report from the output of run_estimators

    Parameters
    ----------
    output_run_estimators : array of dicts
        Array of model dictionaries produced by run_estimators
    digits :  int, opttional
        Number of digits to round output
    print_params : bool, optional
        Enable report to include model parameters

    Returns
    -------
    None
    """
    for i, model in enumerate(output_run_estimators):
        name = model["name"]
        train_score = colored(np.round(model["train_score_mean"], digits), "grey")
        train_sd = colored(np.round(model["train_score_sd"], digits), "grey")
        test_score = colored(np.round(model["test_score_mean"], digits), "green")
        test_sd = colored(np.round(model["test_score_sd"], digits), "green")

        print(f'{i+1}. {name}')
        print(f'\tTrain Score:\t {train_score} +/- {train_sd}')
        print(f'\tTest Score:\t {test_score} +/- {test_sd}')
        if print_params:
            print(model['params'])
        print()


def run_estimators(estimators,
                   X_train,
                   y_train,
                   X_test,
                   y_test,
                   scoring,
                   cv=5,
                   verbose=False):
    """Runs a list of models and returns a list of sorted scores by model

    Parameters
    ----------
    estimators : array_like
        Array of estimators
    X_train : array_like
    y_train : array_like
    X_test : array_like
    y_test : array_like
    scoring : string
        Scikit-Learn scoring string
        See: https://scikit-learn.org/stable/modules/model_evaluation.html
    cv : int, optional
        Number of cross-validation folds
    verbose : bool, optional
        Verbosity flag

    Returns
    -------
    array_like
        List of models sorted by score
    """
    scores = []

    for clf in estimators:
        print(f'Running {clf.__class__.__name__}...')
        start_time = timeit.default_timer()
        clf.fit(X_train, y_train)
        elapsed = timeit.default_timer() - start_time
        print(f'Time Elapsed: {str(elapsed)} seconds')

        train_scores = cross_val_score(clf, X_train, y_train, scoring=scoring, cv=cv)
        test_scores = cross_val_score(clf, X_test, y_test, scoring=scoring, cv=cv)

        mean_train_cv_score = np.mean(train_scores)
        std_def_train_cv_score = np.std(train_scores)
        mean_test_cv_score = np.mean(test_scores)
        std_def_test_cv_score = np.std(test_scores)

        if verbose:
            print(f'Mean Train CV Score:\t {mean_train_cv_score}')
            print(f'SD Train CV Score:\t {std_def_train_cv_score}')
            print(f'Mean Test CV Score:\t {mean_test_cv_score}')
            print(f'SD Test CV Score:\t {std_def_test_cv_score}\n')

        score = {
            'name': clf.__class__.__name__,
            'params': clf.get_params(),
            'train_score_mean': mean_train_cv_score,
            'train_score_sd': std_def_train_cv_score,
            'test_score_mean': mean_test_cv_score,
            'test_score_sd': std_def_test_cv_score
        }
        scores.append(score)

    return sorted(scores, key=lambda x: x['test_score_mean'], reverse=True)

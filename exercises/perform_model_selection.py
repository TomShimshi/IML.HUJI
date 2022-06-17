from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) +
    # eps for eps Gaussian noise and split into training- and testing portions
    eps = np.random.normal(0, noise, n_samples)
    f_x = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    X = np.linspace(-1.2, 2, n_samples)
    y_without_noise = f_x(X)
    y = y_without_noise + eps

    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y), 2.0/3)
    X_train, y_train, X_test, y_test = X_train[0].to_numpy(), y_train.to_numpy(), \
                                       X_test[0].to_numpy(), y_test.to_numpy()


    fig1 = go.Figure([go.Scatter(x=X, y=y_without_noise, mode='lines', name="True Model "
                                                                            "(without noise)"),
                     go.Scatter(x=X_train, y=y_train, mode='markers', name="Train"),
                     go.Scatter(x=X_test, y=y_test, mode='markers', name="Test")])
    fig1.update_layout(height=600, title_text="True Model VS Train & Test Sets")
    fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    deg = 11
    deg_list = np.arange(deg)
    train_score, validation_score = np.zeros(deg), np.zeros(deg)
    for k in deg_list:
        train_score[k], validation_score[k] = cross_validate(PolynomialFitting(k), X_train,
                                                             y_train, mean_square_error, 5)

    fig2 = go.Figure([go.Scatter(x=deg_list, y=train_score, mode='lines', name="Train"),
                      go.Scatter(x=deg_list, y=validation_score, mode='lines', name="Validation")])
    fig2.update_layout(height=600, title_text="Errors for Train and Validation Sets as a Function "
                                              "of The Polynomial Degree")
    fig2.show()


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(validation_score)
    best_model = PolynomialFitting(k_star)
    best_model.fit(X_train, y_train)
    error = mean_square_error(y_test, best_model.predict(X_test))
    print("Best k is- {0}, The Test Error is- {1}".format(k_star, np.round(error, 2)))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting
    regularization parameter values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, y_train, X_test, y_test = \
        X[:n_samples, :], y[:n_samples], X[n_samples:, :], y[n_samples:]


    # Question 7 - Perform CV for different values of the regularization parameter for
    # Ridge and Lasso regressions
    ridge_train_errors, ridge_validation_errors, lasso_train_errors, lasso_validation_errors = \
        np.zeros(n_evaluations), np.zeros(n_evaluations), \
        np.zeros(n_evaluations), np.zeros(n_evaluations),

    lam_list = np.linspace(0.008, 7, n_evaluations)
    for index, lam in enumerate(lam_list):
        ridge_train_errors[index], ridge_validation_errors[index] = cross_validate(
                                        RidgeRegression(lam), X_train, y_train, mean_square_error)
        lasso_train_errors[index], lasso_validation_errors[index] = cross_validate(
                                                Lasso(lam), X_train, y_train, mean_square_error)

    fig3 = go.Figure([go.Scatter(x=lam_list, y=ridge_train_errors, name="Ridge Train"),
                      go.Scatter(x=lam_list, y=ridge_validation_errors, name="Ridge Validation"),
                      go.Scatter(x=lam_list, y=lasso_train_errors, name="Lasso Train"),
                      go.Scatter(x=lam_list, y=lasso_validation_errors, name="Lasso Validation")])
    fig3.update_layout(height=600, title_text="Ridge VS Lasso errors as a Function of Lambda")
    fig3.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lam_ridge = lam_list[np.argmin(ridge_validation_errors)]
    best_ridge = RidgeRegression(best_lam_ridge)
    best_ridge.fit(X_train, y_train)
    error_ridge = mean_square_error(y_test, best_ridge.predict(X_test))
    print("Best lambda for Ridge is- {0}, The Test Error is- {1}".format(
        best_lam_ridge, error_ridge))

    best_lam_lasso = lam_list[np.argmin(lasso_validation_errors)]
    best_lasso = Lasso(best_lam_lasso)
    best_lasso.fit(X_train, y_train)
    error_lasso = mean_square_error(
        y_test, best_lasso.predict(X_test))
    print("Best lambda for Lasso is- {0}, The Test Error is- {1}".format(
        best_lam_lasso, error_lasso))

    best_linear = LinearRegression()
    best_linear.fit(X_train, y_train)
    error_linear = mean_square_error(y_test, best_linear.predict(X_test))
    print("The Test Error for Linear Regression is- {0}".format(error_linear))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()

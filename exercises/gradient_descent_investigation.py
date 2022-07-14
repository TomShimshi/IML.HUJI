import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.metrics.loss_functions import misclassification_error
from IMLearn.model_selection import cross_validate
import plotly.graph_objects as go



def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values_list = []
    weights_list = []

    def callback(**kwargs):
        values_list.append(kwargs["val"])
        weights_list.append(kwargs["weights"])

    return callback, values_list, weights_list


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):

    l1_loss, l1_eta, l2_loss, l2_eta = [], [], [], []

    for ind, modul in enumerate([L1, L2]):
        for eta in etas:
            callback_func, values_list, weights_list = get_gd_state_recorder_callback()
            gd = GradientDescent(FixedLR(eta), callback=callback_func)
            gd.fit(modul(init), X=None, y=None)

            # Q1
            fig1 = plot_descent_path(modul, np.array(weights_list),
                            "{0} Norm Decent Trajectory for eta = {1}".format(modul.__name__, eta))

            fig1.show()

            # Q2
            fig2 = go.Figure(go.Scatter(x=list(range(0, len(values_list))), y=values_list,
                                        mode="lines+markers", name=modul.__name__))
            fig2.update_layout(title="Convergence Rate for {0} Norm and eta = {1}".
                               format(modul.__name__, eta),
                               xaxis={"title": "Num of Iteration"},
                               yaxis={"title": "Norm"})
            fig2.show()

            if (ind == 0):
                l1_loss.append(np.min(values_list))
                l1_eta.append(eta)

            else:
                l2_loss.append(np.min(values_list))
                l2_eta.append(eta)

    l1_min_loss, l2_min_loss = np.min(l1_loss), np.min(l2_loss)
    l1_min_eta, l2_min_eta = l1_eta[np.argmin(l1_loss)], l2_eta[np.argmin(l2_loss)]

    print("The lowest loss recorded for modul L1 is- {0}, and the eta is- {1}".format(l1_min_loss,
                                                                                      l1_min_eta))
    print("The lowest loss recorded for modul L2 is- {0}, and the eta is- {1}".format(l2_min_loss,
                                                                                      l2_min_eta))



def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):

    # Optimize the L1 objective using different decay-rate values of the exponentially
    # decaying learning rate

    # Plot algorithm's convergence for the different values of gamma
    min_norm = np.inf

    fig3 = go.Figure().update_layout(title="Convergence for L1 Norm as a Function of Decay-Rate",
                       xaxis={"title": "Num of Iteration"},
                       yaxis={"title": "Norm"})

    for gamma in gammas:
        callback_func, values_list, weights_list = get_gd_state_recorder_callback()
        gd = GradientDescent(ExponentialLR(eta, gamma), callback=callback_func)
        gd.fit(L1(init), X=None, y=None)

        curr_min = np.min(values_list)
        if curr_min < min_norm:
            min_norm = curr_min

        fig3.add_trace(go.Scatter(x=list(range(0, len(values_list))), y=values_list,
                                    mode="lines+markers", name="gamma = {0}".format(gamma)))

    fig3.show()

    print("The lowest loss recorded for L1 is- {0}".format(min_norm))

    # Plot descent path for gamma=0.95
    for ind, modul in enumerate([L1, L2]):
        callback_func, values_list, weights_list = get_gd_state_recorder_callback()
        gd = GradientDescent(ExponentialLR(eta, .95), callback=callback_func)
        gd.fit(modul(init), X=None, y=None)

        fig4 = plot_descent_path(modul, np.array(weights_list),
                         "{0} Norm Decent Trajectory for gamma = 0.95".format(modul.__name__))

        fig4.show()



def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    from sklearn.metrics import roc_curve

    lr = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000))
    lr.fit(np.array(X_train), np.array(y_train))
    y_proba = lr.predict_proba(np.array(X_train))
    fpr, tpr, thresh = roc_curve(y_train, y_proba)
    fig5 = go.Figure(
        go.Scatter(x=fpr, y=tpr, mode='lines+markers', text=thresh, name="",
                   hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}"))
    fig5.update_layout(title="ROC Curve for Fitted Model- Logistic Regressor with Gradient Descent",
                       xaxis={"title": "FPR"},
                       yaxis={"title": "TPR"})
    fig5.show()


    alphas = np.linspace(0, 1, 101)
    alphas_arr = []
    for alpha in alphas:
        y_prob_alpha = np.where(y_proba >= alpha, 1, 0)
        curr_fpr, curr_tpr, curr_thresh = roc_curve(y_train, y_prob_alpha)
        alphas_arr.append((curr_tpr - curr_fpr)[1])

    best_alpha = alphas[np.argmax(alphas_arr)]
    best_alpha_model = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4),
                                                           max_iter=20000), alpha=best_alpha)
    best_alpha_model.fit(np.array(X_train), np.array(y_train))
    best_alpha_error = best_alpha_model.loss(np.array(X_train), np.array(y_train))

    print("The best alpha is- {0}, and the error is- {1}".format(best_alpha, best_alpha_error))


    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # using cross-validation to specify values of regularization parameter l1
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for modul in ['l1', 'l2']:
        validation_res = []
        for lam in lambdas:
            model = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4),
                                                              max_iter=20000),
                                       alpha=0.5, penalty=modul, lam=lam)
            curr_validation = cross_validate(model, np.array(X_train), np.array(y_train),
                                                    misclassification_error)[1]
            validation_res.append(curr_validation)

        best_lam = lambdas[np.argmin(validation_res)]
        best_model = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(1e-4),
                                                               max_iter=20000),
                                        alpha=0.5, penalty=modul, lam=best_lam)
        best_model.fit(np.array(X_train), np.array(y_train))
        best_error = best_model.loss(np.array(X_train), np.array(y_train))
        print("The error recorded in modul {0} is- {1}, for lambda- {2}".format(
                                        modul, best_error, best_lam))


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()


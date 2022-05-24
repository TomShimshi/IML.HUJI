import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y

def decision_surface(t, predict, xrange, yrange, density=120, dotted=False, colorscale=custom, showscale=True):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()], t)

    if dotted:
        return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers", marker=dict(color=pred, size=1, colorscale=colorscale, reversescale=False), hoverinfo="skip", showlegend=False)
    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape), colorscale=colorscale, reversescale=False, opacity=.7, connectgaps=True, hoverinfo="skip", showlegend=False, showscale=showscale)


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):

    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), \
                                           generate_data(test_size, noise)


    # Question 1: Train- and test errors of AdaBoost in noiseless case
    train_errors, test_errors, learners_list = [], [], np.arange(1, n_learners)
    adaBoost = AdaBoost(DecisionStump, n_learners)
    adaBoost.fit(train_X, train_y)

    for learner in learners_list:
        train_errors.append(adaBoost.partial_loss(train_X, train_y, learner))
        test_errors.append(adaBoost.partial_loss(test_X, test_y, learner))

    fig = make_subplots()
    fig.add_traces([go.Scatter(x=learners_list, y=train_errors, mode='lines',
                                   marker=dict(color='blue', line_width=1), name="Train Error"),
                    go.Scatter(x=learners_list, y=test_errors, mode='lines',
                                   marker=dict(color='orange', line_width=1), name="Test Error")])
    fig.update_layout(xaxis_title="Ensemble Size", yaxis_title="Error Rate",
                      title="Q1 - Adaboost Error",
                      font=dict(family="Arial", size=20))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array(
        [np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    fig1 = make_subplots(rows=2, cols=2,
                         subplot_titles=[f"Ensemble Size of {m}" for m in T],
                         horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig1.add_traces([decision_surface(t, adaBoost.fit(train_X, train_y).partial_predict,
                                          lims[0], lims[1], showscale=False),
                         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                                    showlegend=False,
                                    marker=dict(color=train_y,
                                                colorscale=[custom[0], custom[-1]],
                                                line=dict(color="black", width=1)))],
                        rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig1.update_layout(title="Q2 - Decision Surface of Different Ensemble Size",
                       font=dict(family="Arial", size=20),
                       margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig1.show()

    # Question 3: Decision surface of best performing ensemble
    test_error = []
    for T in learners_list:
        test_error.append(adaBoost.partial_loss(test_X, test_y, T))
    T_hat = np.argmin(test_error) + 1

    ensemble_size = (T_hat)
    accuracy = 1 - test_error[T_hat]

    fig2 = make_subplots()
    fig2.add_traces(
        [decision_surface(T_hat, adaBoost.fit(train_X, train_y).partial_predict, lims[0], lims[1],
                          showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=test_y,
                                colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1)))],
    )

    fig2.update_layout(title=f"Q3 - Decision Surface of Ensemble Size = {ensemble_size}, Accuracy = {accuracy}",
                       font=dict(family="Arial", size=20),
                       margin=dict(t=50))
    fig2.show()

    # Question 4: Decision surface with weighted samples
    D = adaBoost.D_
    D = D / np.max(D) * 5
    fig3 = make_subplots()
    fig3.add_traces(
        [decision_surface(T_hat, adaBoost.fit(train_X, train_y).partial_predict, lims[0], lims[1],
                          showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=train_y,
                                colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1), size=D))],
    )

    fig3.update_layout(title="Q4 - Decision Surface of Ensemble Size = 250, Size of Markers Proportional to Sample Weight",
                       font=dict(family="Arial", size=20),
                       margin=dict(t=100))

    fig3.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)

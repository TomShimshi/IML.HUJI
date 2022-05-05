from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers.
    File is assumed to be an ndarray of shape (n_samples, 3) where the first 2 columns represent
    features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and
    inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable",
                                                                    "linearly_inseparable.npy")]:
        # Load dataset
        data, labels = load_dataset("../datasets/{0}".format(f))

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perceptron = Perceptron(callback=lambda p, xi, yi: losses.append(p.loss(data, labels)))
        perceptron.fit(data, labels)

        # Plot figure of loss as function of fitting iteration
        fig = px.scatter(x=range(1, len(losses)+1), y=losses)
        fig.update_traces(mode="lines")
        fig.update_layout(xaxis_title="Iterations",
                          yaxis_title="Loss Recorded",
                          title="Loss progression of the Perceptron \
                                algorithm over {0} dataset".format(n),
                          font=dict(family="Arial", size=18))
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] <
                                                                                  cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    from IMLearn.metrics import accuracy

    symbols = np.array(["circle", "diamond", "cross"])
    colors = np.array(["LightSkyBlue", "MediumPurple", "MidnightBlue"])

    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data, true_labels = load_dataset("../datasets/{0}".format(f))
        # Fit models and predict over training set
        lda = LDA()
        lda.fit(data, true_labels)
        lda_y = lda.predict(data)
        lda_acc = accuracy(true_labels, lda_y)

        gnb = GaussianNaiveBayes()
        gnb.fit(data, true_labels)
        gnb_y = gnb.predict(data)
        gnb_acc = accuracy(true_labels, gnb_y)

        means_lda = np.array(lda.mu_)
        means_gnb = np.array(gnb.mu_)

        # Plot a figure with two subplots, showing the Gaussian Naive Bayes predictions on the left
        # and LDA predictions on the right. Plot title should specify dataset used and
        # subplot titles should specify algorithm and accuracy
        # Create subplots

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("LDA Classifier, accuracy: {:.3f}".format(lda_acc),
                                            "Naive Bayes Classifier, accuracy: {:.3f}".format(
                                                gnb_acc)))

        fig.add_traces([go.Scatter(x=data[:, 0], y=data[:, 1], mode='markers')], rows=1, cols=1)
        fig.add_traces([go.Scatter(x=data[:, 0], y=data[:, 1], mode='markers')], rows=1, cols=2)

        # Add traces for data-points setting symbols and colors
        fig.update_traces(marker=dict(size=10, color=colors[lda_y.astype(int)],
                                                line=dict(width=.5),
                                               symbol=symbols[true_labels.astype(int)]))

        fig.update_traces(marker=dict(size=10, color=colors[gnb_y.astype(int)],
                                      line=dict(width=.5),
                                               symbol=symbols[true_labels.astype(int)]))

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_traces([go.Scatter(x=means_lda[:, 0],
                                   y=means_lda[:, 1],
                                   marker=dict(size=10, color='black', symbol='x',
                                               line=dict(width=.5)),
                                   mode='markers')],
                       rows=1, cols=1)
        fig.add_traces([go.Scatter(x=means_gnb[:, 0],
                                   y=means_gnb[:, 1],
                                   marker=dict(size=10, color='black', symbol='x',
                                               line=dict(width=.5)),
                                   mode='markers')],
                       rows=1, cols=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(lda.classes_)):
            fig.add_trace(get_ellipse(means_lda[i, :], lda.cov_), row=1, col=1)

        for i in range(len(gnb.classes_)):
            fig.add_trace(get_ellipse(means_gnb[i, :], np.diag(gnb.vars_[i])), row=1, col=2)

        fig.update_layout(
            title="Performance of Classification for {0} dataset".format(f), showlegend=False)

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
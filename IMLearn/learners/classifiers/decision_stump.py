from __future__ import annotations
from typing import Tuple, NoReturn
from IMLearn.base.base_estimator import BaseEstimator
# from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        best_loss = None
        for sign, j in product([-1, 1], range(X.shape[1])):
            loss, thr = self._find_threshold(X[:, j], y, sign)
            if best_loss is None or loss < best_loss:
                self.sign_, self.threshold_, self.j_ = sign, thr, j
                best_loss = loss

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return self.sign_ * ((X[:, self.j_] >= self.threshold_) * 2 - 1)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sort_idx = np.argsort(values)
        values, labels, D = values[sort_idx], np.sign(labels)[sort_idx], np.abs(labels)[sort_idx]
        thetas = np.concatenate([[-np.inf], (values[1:] + values[:-1]) / 2, [np.inf]])
        minimal_theta_loss = np.sum(D[labels == sign])
        losses = np.append(minimal_theta_loss, minimal_theta_loss - np.cumsum(D * (labels * sign)))
        min_loss_idx = np.argmin(losses)
        return losses[min_loss_idx], thetas[min_loss_idx]

        # min_loss = 0
        # curr_loss = 0
        # min_idx = 0
        # for i in range(values.shape[0]):
        #     y_pred = np.zeros(values.shape[0])
        #     for j in range(values.shape[0]):
        #         if values[j] >= values[i]:
        #             y_pred[j] = sign
        #         else:
        #             y_pred[j] = -sign
        #     curr_loss = self._loss(labels, y_pred)
        #     if curr_loss < min_loss:
        #         min_loss = curr_loss
        #         min_idx = i
        #
        # return values[min_idx], min_loss

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X)
        return float(np.sum(np.sign(y) != np.sign(y_pred)))

"""Multinomial Naive Bayes written from scratch."""
import numpy as np


class NaiveBayesClassifier:
    """Multinomial Naive Bayes with Laplace (add-one) smoothing.

    Works with dense arrays or scipy sparse matrices of token counts.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        y = np.asarray(y)
        # np.unique returns sorted classes, so priors and likelihoods share one ordering.
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_log_prior_ = np.log(counts / len(y))

        # Total count of each feature (token) within each class.
        class_feature_counts = np.vstack([
            np.asarray(X[y == cls].sum(axis=0)).ravel() for cls in self.classes_
        ])

        smoothed = class_feature_counts + self.alpha
        self.feature_log_prob_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        return self

    def predict_log_proba(self, X) -> np.ndarray:
        """Unnormalised joint log-likelihood log P(x, c) for each sample and class."""
        return np.asarray(X @ self.feature_log_prob_.T) + self.class_log_prior_

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

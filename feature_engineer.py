# feature_engineer.py
# Transforms the raw 4-feature matrix into a richer representation.
#
# WHY polynomial features?
#   Linear regression fits:  y = w0 + w1*x1 + w2*x2 + ...
#   This is a STRAIGHT LINE in feature space.
#   But bottleneck % has a non-linear relationship with benchmark scores —
#   doubling the GPU score doesn't halve the bottleneck linearly.
#
#   By adding x1², x2², x1*x2 etc. as NEW columns, we let the
#   linear model fit a CURVE in the original space.
#   The model is still "linear in the weights" — just richer inputs.
#
# OOP concept: Open/Closed Principle
#   Preprocessor is closed for modification but open for extension.
#   We add features without touching the existing pipeline.

import numpy as np
from itertools import combinations_with_replacement
from logger import Logger


class FeatureEngineer:
    """
    Builds polynomial and interaction features from the base feature matrix.

    degree=1  →  original 4 features  (same as before)
    degree=2  →  4 original + 10 squared/cross terms = 14 features
    degree=3  →  4 original + 10 degree-2 + 20 degree-3 = 34 features

    For each pair (xi, xj) with i <= j, we add xi * xj.
    For each triple, we add xi * xj * xk.  And so on.
    """

    def __init__(self, degree: int = 2):
        self.degree      = degree
        self.log         = Logger("FeatureEngineer")
        self.n_input     = None   # number of original features
        self.combinations = None  # index tuples for each new feature

    def fit(self, X: np.ndarray):
        """
        Learns which index combinations to multiply together.
        Must be called on training data ONLY — then transform()
        applies the same expansion to test data and new inputs.
        """
        self.n_input = X.shape[1]

        # combinations_with_replacement([0,1,2,3], 2) gives:
        # (0,0),(0,1),(0,2),(0,3),(1,1),(1,2),(1,3),(2,2),(2,3),(3,3)
        # Each tuple (i,j) means "multiply feature i by feature j"
        self.combinations = []
        for d in range(2, self.degree + 1):
            self.combinations += list(
                combinations_with_replacement(range(self.n_input), d)
            )

        n_new     = len(self.combinations)
        n_total   = self.n_input + n_new
        self.log.info(f"Degree        : {self.degree}")
        self.log.info(f"Original cols : {self.n_input}")
        self.log.info(f"New poly cols : {n_new}")
        self.log.info(f"Total features: {n_total}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Appends polynomial/interaction columns to X.
        Returns new array — does NOT modify X in place.
        """
        new_cols = []
        for combo in self.combinations:
            # Multiply the columns at the indices in combo together
            # e.g. combo=(0,1) → X[:,0] * X[:,1]
            # e.g. combo=(2,2) → X[:,2] ** 2
            col = np.ones(X.shape[0])
            for idx in combo:
                col = col * X[:, idx]
            new_cols.append(col.reshape(-1, 1))

        X_poly = np.hstack([X] + new_cols)   # (m, n_original + n_new)
        return X_poly

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit then transform in one call."""
        return self.fit(X).transform(X)

    def get_feature_count(self) -> int:
        return self.n_input + len(self.combinations)

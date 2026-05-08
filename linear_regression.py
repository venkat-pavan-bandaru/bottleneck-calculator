# linear_regression.py
# The model. Pure NumPy — no sklearn anywhere.
# OOP concepts:
#   Encapsulation  — weights and bias are private state
#   Abstraction    — caller just uses .fit() and .predict()

import numpy as np
from config import Config
from logger import Logger


class LinearRegression:
    """
    Multivariate linear regression trained with batch gradient descent.

    The model:
        ŷ = X·w + b
        where X  shape (m, n)   — design matrix (m samples, n features)
              w  shape (n,)     — weight vector  (one per feature)
              b  scalar         — bias term

    Cost function (MSE):
        J(w, b) = (1 / 2m) * sum( (ŷ - y)² )

    Gradients (partial derivatives):
        ∂J/∂w = (1/m) * Xᵀ · (ŷ - y)    shape (n,)
        ∂J/∂b = (1/m) * sum(ŷ - y)       scalar

    Update rule (gradient descent):
        w ← w - α * ∂J/∂w
        b ← b - α * ∂J/∂b
    """

    def __init__(self):
        self.log    = Logger("LinearRegression")
        self.w      = None       # weights  shape (n,)
        self.b      = 0.0        # bias     scalar
        self.costs  = []         # cost history, one per iteration

    # ─────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray,
            lr: float = None, iterations: int = None):
        """
        Runs batch gradient descent until we hit `iterations`.

        Parameters
        ----------
        X          : training features, shape (m, n)
        y          : training targets,  shape (m,)
        lr         : learning rate α  (defaults to Config value)
        iterations : number of steps   (defaults to Config value)
        """
        lr         = lr         or Config.LEARNING_RATE
        iterations = iterations or Config.ITERATIONS

        m, n = X.shape   # m = samples, n = features

        # Initialise weights to zero
        # (random init also works; zero is fine for linear regression)
        self.w = np.zeros(n)
        self.b = 0.0
        self.costs = []

        self.log.section("Training")
        self.log.info(f"Samples      : {m:,}")
        self.log.info(f"Features     : {n}")
        self.log.info(f"Learning rate: {lr}")
        self.log.info(f"Iterations   : {iterations}")
        self.log.info("")

        for i in range(1, iterations + 1):

            # ── Forward pass ──────────────────────────────
            # ŷ = X·w + b          shape (m,)
            y_hat = self._predict_raw(X)

            # ── Compute cost ──────────────────────────────
            # J = (1/2m) * ||ŷ - y||²
            cost = self._cost(y_hat, y, m)
            self.costs.append(cost)

            # ── Compute gradients ─────────────────────────
            # error = ŷ - y        shape (m,)
            # dw    = (1/m) Xᵀ·error   shape (n,)
            # db    = (1/m) sum(error)  scalar
            error = y_hat - y
            dw    = (1.0 / m) * X.T @ error
            db    = (1.0 / m) * np.sum(error)

            # ── Update weights ────────────────────────────
            self.w -= lr * dw
            self.b -= lr * db

            # ── Log every N iterations ────────────────────
            if i % Config.LOG_EVERY_N_ITERS == 0 or i == 1:
                self.log.cost(i, cost)

        self.log.success(f"Final cost   : {self.costs[-1]:.6f}")
        self.log.success(f"Cost reduced : {self.costs[0]:.4f}  →  {self.costs[-1]:.4f}")
        return self

    # ─────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns predicted bottleneck % for each row in X.
        X must already be scaled with the same mean/std used in fit().
        """
        if self.w is None:
            raise RuntimeError("Model is not trained yet. Call .fit() first.")
        return self._predict_raw(X)

    # ─────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        # ŷ = X·w + b
        return X @ self.w + self.b

    @staticmethod
    def _cost(y_hat: np.ndarray, y: np.ndarray, m: int) -> float:
        # J = (1/2m) * sum( (ŷ - y)² )
        return float(np.sum((y_hat - y) ** 2) / (2 * m))

    # ─────────────────────────────────────────────────────
    # Inspection
    # ─────────────────────────────────────────────────────
    def get_weights(self):
        return self.w, self.b

    def get_cost_history(self):
        return self.costs

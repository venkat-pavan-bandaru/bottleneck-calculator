# ridge_regression.py
# Ridge Regression = Linear Regression + L2 regularisation penalty.
#
# WHY Ridge?
#   When we add polynomial features (degree=2 → 14 features),
#   the model can OVERFIT — it memorises training noise instead
#   of learning real patterns.  It fits training data perfectly
#   but performs poorly on unseen data.
#
#   Ridge prevents this by adding a penalty term to the cost:
#
#   J(w) = (1/2m) * sum((ŷ - y)²)  +  (λ/2m) * sum(w²)
#                                        ↑
#                                 L2 penalty — punishes large weights
#
#   Large weights → complex model → overfitting.
#   The penalty forces weights to stay small → simpler model → generalises.
#
#   λ (lambda) controls the trade-off:
#     λ = 0   →  plain linear regression (no regularisation)
#     λ large →  all weights pushed toward zero (underfitting)
#     λ sweet spot → best generalisation
#
# OOP concept: Inheritance-ready design
#   Ridge shares the same interface as LinearRegression (.fit, .predict)
#   so ModelComparison can treat both identically.

import numpy as np
from config import Config
from logger import Logger


class RidgeRegression:
    """
    Multivariate Ridge Regression trained with batch gradient descent.

    Cost:       J(w,b) = (1/2m)*||Xw+b - y||² + (λ/2m)*||w||²
    Gradient w: ∂J/∂w  = (1/m)*Xᵀ(Xw+b - y)  + (λ/m)*w
    Gradient b: ∂J/∂b  = (1/m)*sum(Xw+b - y)   ← bias NOT regularised
    Update:     w ← w - α*(∂J/∂w)
                b ← b - α*(∂J/∂b)

    Note: bias b is intentionally NOT regularised.
    Regularising the bias shifts all predictions uniformly — that
    doesn't reduce model complexity, it just introduces bias.
    """

    def __init__(self, lambda_: float = 1.0):
        """
        Parameters
        ----------
        lambda_ : regularisation strength (λ).
                  Underscore avoids clash with Python's 'lambda' keyword.
        """
        self.lambda_ = lambda_
        self.log     = Logger("RidgeRegression")
        self.w       = None
        self.b       = 0.0
        self.costs   = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            lr: float = None, iterations: int = None):

        lr         = lr         or Config.LEARNING_RATE
        iterations = iterations or Config.ITERATIONS

        m, n = X.shape
        self.w     = np.zeros(n)
        self.b     = 0.0
        self.costs = []

        self.log.section("Training Ridge Regression")
        self.log.info(f"Samples      : {m:,}")
        self.log.info(f"Features     : {n}")
        self.log.info(f"Learning rate: {lr}")
        self.log.info(f"Iterations   : {iterations}")
        self.log.info(f"Lambda (λ)   : {self.lambda_}")
        self.log.info("")

        for i in range(1, iterations + 1):

            # ── Forward pass ──────────────────────────────
            y_hat = X @ self.w + self.b        # shape (m,)

            # ── Cost with L2 penalty ─────────────────────
            mse_term     = np.sum((y_hat - y) ** 2) / (2 * m)
            penalty_term = (self.lambda_ / (2 * m)) * np.sum(self.w ** 2)
            cost         = mse_term + penalty_term
            self.costs.append(cost)

            # ── Gradients ─────────────────────────────────
            error = y_hat - y                          # shape (m,)
            dw    = (X.T @ error) / m + (self.lambda_ / m) * self.w
            #                           ↑
            #                    extra penalty gradient
            db    = np.sum(error) / m                  # bias: no penalty

            # ── Update ────────────────────────────────────
            self.w -= lr * dw
            self.b -= lr * db

            if i % Config.LOG_EVERY_N_ITERS == 0 or i == 1:
                self.log.cost(i, cost)

        self.log.success(f"Final cost   : {self.costs[-1]:.6f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Model not trained. Call .fit() first.")
        return X @ self.w + self.b

    def get_weights(self):
        return self.w, self.b

    def get_cost_history(self):
        return self.costs

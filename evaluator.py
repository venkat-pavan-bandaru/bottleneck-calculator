# evaluator.py
# Responsible for: computing evaluation metrics from scratch.
# OOP concept: Abstraction — main.py doesn't know the formulas,
#              it just calls .evaluate() and reads the results.

import numpy as np
from logger import Logger


class Evaluator:
    """
    Computes regression metrics without sklearn.

    Metrics implemented:
        MSE   Mean Squared Error       = (1/m) * sum((ŷ - y)²)
        RMSE  Root Mean Squared Error  = sqrt(MSE)
        MAE   Mean Absolute Error      = (1/m) * sum(|ŷ - y|)
        R²    Coefficient of Determination

    R² formula:
        SS_res = sum( (y - ŷ)² )         residual sum of squares
        SS_tot = sum( (y - ȳ)² )         total sum of squares
        R²     = 1 - (SS_res / SS_tot)

        R² = 1.0 → perfect fit
        R² = 0.0 → model predicts the mean every time (useless)
        R² < 0   → model is worse than guessing the mean
    """

    def __init__(self):
        self.log     = Logger("Evaluator")
        self.results = {}

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, label: str = "Test"):
        """
        Computes all metrics and prints a summary.

        Parameters
        ----------
        y_true : actual target values
        y_pred : model predictions
        label  : prefix for the log output (e.g. "Train" or "Test")
        """
        self.log.section(f"Evaluation — {label} Set")

        mse  = self._mse(y_true, y_pred)
        rmse = self._rmse(mse)
        mae  = self._mae(y_true, y_pred)
        r2   = self._r2(y_true, y_pred)

        self.results = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

        self.log.info(f"MSE   = {mse:.4f}")
        self.log.info(f"RMSE  = {rmse:.4f}  (± {rmse:.2f}% bottleneck error on average)")
        self.log.info(f"MAE   = {mae:.4f}")
        self.log.info(f"R²    = {r2:.4f}  {'✓ strong fit' if r2 > 0.85 else '△ moderate fit' if r2 > 0.6 else '✗ weak fit'}")

        return self   # allow chaining

    # ─────────────────────────────────────────────────────
    # Metric formulas — all private, all pure numpy
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _mse(y_true, y_pred) -> float:
        # (1/m) * sum( (ŷ - y)² )
        return float(np.mean((y_pred - y_true) ** 2))

    @staticmethod
    def _rmse(mse: float) -> float:
        return float(np.sqrt(mse))

    @staticmethod
    def _mae(y_true, y_pred) -> float:
        # (1/m) * sum( |ŷ - y| )
        return float(np.mean(np.abs(y_pred - y_true)))

    @staticmethod
    def _r2(y_true, y_pred) -> float:
        y_mean  = np.mean(y_true)
        ss_res  = np.sum((y_true - y_pred)  ** 2)   # residual
        ss_tot  = np.sum((y_true - y_mean)  ** 2)   # total variance
        return float(1 - ss_res / ss_tot)

    def get_results(self) -> dict:
        return self.results

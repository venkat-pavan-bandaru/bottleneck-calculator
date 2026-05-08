# cross_validator.py
# K-Fold Cross Validation from scratch.
#
# WHY cross validation?
#   A single train/test split is LUCKY or UNLUCKY depending on which
#   rows ended up in each set.  If all easy examples land in the test
#   set, R² looks great but the model is actually weak.
#
#   K-fold solves this by rotating the test set:
#     Split data into K equal "folds" (chunks).
#     Round 1: train on folds 2-5, test on fold 1
#     Round 2: train on folds 1,3-5, test on fold 2
#     ...
#     Round K: train on folds 1-4, test on fold 5
#
#   Every sample is tested EXACTLY ONCE.
#   Final score = mean R² across all K rounds.
#   This is the gold standard for model evaluation.
#
# OOP concept: Dependency Inversion
#   CrossValidator doesn't care which model it receives —
#   it just calls .fit() and .predict() on whatever is passed in.
#   Works with LinearRegression, RidgeRegression, or any future model.

import numpy as np
from logger import Logger


class CrossValidator:
    """
    K-Fold cross validation. Works with any model that
    implements .fit(X, y) and .predict(X).
    """

    def __init__(self, k: int = 5):
        """
        Parameters
        ----------
        k : number of folds. 5 is the standard default.
            More folds = more reliable estimate, but slower.
        """
        self.k   = k
        self.log = Logger("CrossValidator")

    def validate(self, model_class, X: np.ndarray, y: np.ndarray,
                 model_kwargs: dict = None, name: str = "Model") -> dict:
        """
        Runs K-fold CV and returns mean/std of metrics.

        Parameters
        ----------
        model_class  : the class itself (e.g. RidgeRegression)
                       A fresh instance is created per fold.
        X            : full feature matrix (already scaled)
        y            : full target vector
        model_kwargs : dict of constructor args, e.g. {"lambda_": 0.1}
        name         : label for logging

        Returns
        -------
        dict with keys: r2_scores, mean_r2, std_r2, mean_rmse
        """
        model_kwargs = model_kwargs or {}
        self.log.section(f"K-Fold CV ({self.k} folds) — {name}")

        m          = len(X)
        fold_size  = m // self.k
        r2_scores  = []
        rmse_scores = []

        # Shuffle indices once — same shuffle for every fold comparison
        np.random.seed(42)
        shuffled = np.random.permutation(m)

        for fold in range(self.k):

            # ── Build test indices for this fold ──────────
            test_start = fold * fold_size
            test_end   = test_start + fold_size if fold < self.k - 1 else m
            # Last fold absorbs any remainder so no data is wasted

            test_idx  = shuffled[test_start : test_end]
            train_idx = np.concatenate([
                shuffled[:test_start],
                shuffled[test_end:]
            ])

            X_train_f, y_train_f = X[train_idx], y[train_idx]
            X_test_f,  y_test_f  = X[test_idx],  y[test_idx]

            # ── Fresh model each fold ─────────────────────
            # IMPORTANT: must be a NEW instance per fold.
            # Reusing the same instance would continue training
            # from the previous fold's weights — data leakage.
            model = model_class(**model_kwargs)
            model.fit(X_train_f, y_train_f, lr=0.1, iterations=3000)

            # ── Score this fold ───────────────────────────
            y_pred = model.predict(X_test_f)
            r2     = self._r2(y_test_f, y_pred)
            rmse   = self._rmse(y_test_f, y_pred)

            r2_scores.append(r2)
            rmse_scores.append(rmse)

            self.log.info(f"  Fold {fold+1}/{self.k}  R²={r2:.4f}  RMSE={rmse:.4f}")

        mean_r2   = float(np.mean(r2_scores))
        std_r2    = float(np.std(r2_scores))
        mean_rmse = float(np.mean(rmse_scores))

        self.log.success(f"Mean R²  : {mean_r2:.4f}  ±{std_r2:.4f}")
        self.log.success(f"Mean RMSE: {mean_rmse:.4f}")

        return {
            "r2_scores"  : r2_scores,
            "mean_r2"    : mean_r2,
            "std_r2"     : std_r2,
            "mean_rmse"  : mean_rmse,
            "rmse_scores": rmse_scores,
        }

    # ─────────────────────────────────────────────────────
    # Private metric helpers
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _r2(y_true, y_pred) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot)

    @staticmethod
    def _rmse(y_true, y_pred) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

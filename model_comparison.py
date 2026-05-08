# model_comparison.py
# Trains all model candidates, collects metrics, picks the winner.
# OOP concept: Strategy Pattern — each model is a swappable strategy.

import numpy as np
from logger import Logger
from linear_regression import LinearRegression
from ridge_regression  import RidgeRegression
from config import Config


class ModelComparison:

    def __init__(self):
        self.log     = Logger("ModelComparison")
        self.results = []
        self.best    = None

    def run(self, X_train, y_train, X_test, y_test):
        self.log.section("Model Comparison")

        lr   = Config.LEARNING_RATE
        itr  = Config.ITERATIONS

        candidates = [
            ("Linear Regression",  LinearRegression(),              lr,    itr),
            ("Ridge  λ=0.01",      RidgeRegression(lambda_=0.01),   lr,    itr),
            ("Ridge  λ=0.1",       RidgeRegression(lambda_=0.1),    lr,    itr),
            ("Ridge  λ=1.0",       RidgeRegression(lambda_=1.0),    lr,    itr),
            ("Ridge  λ=10.0",      RidgeRegression(lambda_=10.0),   lr,    itr),
        ]

        for label, model, lr_, itr_ in candidates:
            model.fit(X_train, y_train, lr=lr_, iterations=itr_)

            train_pred = model.predict(X_train)
            test_pred  = model.predict(X_test)

            result = {
                "label"     : label,
                "model"     : model,
                "train_r2"  : self._r2(y_train, train_pred),
                "test_r2"   : self._r2(y_test,  test_pred),
                "test_rmse" : self._rmse(y_test, test_pred),
                "test_mae"  : self._mae(y_test,  test_pred),
            }
            result["overfit_gap"] = result["train_r2"] - result["test_r2"]
            self.results.append(result)

        self.results.sort(key=lambda r: r["test_r2"], reverse=True)
        self._print_table()
        self.best = self.results[0]["model"]
        self.log.success(f"Winner: {self.results[0]['label']}")
        return self

    def _print_table(self):
        print()
        hdr = f"  {'Model':<22} {'Train R²':>9} {'Test R²':>9} {'RMSE':>8} {'MAE':>8} {'Overfit':>8}"
        print(hdr)
        print("  " + "─" * 60)
        for r in self.results:
            flag = "  ← best" if r is self.results[0] else ""
            print(f"  {r['label']:<22}  {r['train_r2']:>8.4f}  {r['test_r2']:>8.4f}"
                  f"  {r['test_rmse']:>8.4f}  {r['test_mae']:>8.4f}  {r['overfit_gap']:>8.4f}{flag}")
        print()

    def get_best_model(self): return self.best
    def get_results(self):    return self.results

    @staticmethod
    def _r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return float(1 - ss_res / ss_tot)

    @staticmethod
    def _rmse(y_true, y_pred):
        return float(np.sqrt(np.mean((y_true - y_pred)**2)))

    @staticmethod
    def _mae(y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

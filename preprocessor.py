# preprocessor.py
# Responsible for: building the dataset from raw CPU/GPU tables,
#                  engineering the target, scaling features, splitting data.
# OOP concepts:
#   Encapsulation  — all the "how" (formulas, indices) is hidden inside methods
#   Abstraction    — main.py just calls .run() and gets back clean arrays

import numpy as np
import pandas as pd
from config import Config
from logger import Logger


class Preprocessor:

    def __init__(self, cpu_df: pd.DataFrame, gpu_df: pd.DataFrame):
        self.cpu_df = cpu_df
        self.gpu_df = gpu_df
        self.log    = Logger("Preprocessor")

        # Will be populated after run()
        self.X_train = None
        self.X_test  = None
        self.y_train = None
        self.y_test  = None
        self.X_mean  = None
        self.X_std   = None

        # Saved so Predictor can reuse the exact same normalisation
        self._cpu_min = None
        self._cpu_max = None
        self._gpu_min = None
        self._gpu_max = None

    # ─────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────
    def run(self):
        self.log.section("Preprocessing")

        df = self._cross_join()
        df = self._engineer_target(df)
        X, y = self._build_features(df)
        X_scaled = self._standardise(X)
        self._split(X_scaled, y)

        self.log.success("Preprocessing complete")
        return self   # allow chaining

    # ─────────────────────────────────────────────────────
    # Private helpers  (convention: leading _ = internal)
    # ─────────────────────────────────────────────────────

    def _cross_join(self) -> pd.DataFrame:
        """
        Pairs every CPU with every GPU.
        1,114 CPUs × 289 GPUs = ~321,946 possible PC builds.
        Each row represents one build configuration.
        """
        cpu = self.cpu_df.copy()
        gpu = self.gpu_df.copy()
        cpu["_key"] = 1
        gpu["_key"] = 1
        df = pd.merge(cpu, gpu, on="_key").drop(columns="_key")
        self.log.info(f"Cross join    : {len(df):,} build combinations")
        return df

    def _engineer_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives bottleneck_percent from benchmark scores.

        Formula:
            bottleneck = 100 * (gpu_norm - cpu_norm) / (gpu_norm + cpu_norm)

        Range: [-100, +100]
            +100 → pure GPU bottleneck  (GPU crushes CPU)
            -100 → pure CPU bottleneck  (CPU is the weak link)
               0 → balanced
        """
        # Min-max normalise to [0,1] so scores are on the same scale
        self._cpu_min = df["cpuMark"].min()
        self._cpu_max = df["cpuMark"].max()
        self._gpu_min = df["G3Dmark"].min()
        self._gpu_max = df["G3Dmark"].max()

        df["cpu_norm"] = (df["cpuMark"] - self._cpu_min) / (self._cpu_max - self._cpu_min)
        df["gpu_norm"] = (df["G3Dmark"] - self._gpu_min) / (self._gpu_max - self._gpu_min)

        df["bottleneck_percent"] = (
            100.0 * (df["gpu_norm"] - df["cpu_norm"])
            / (df["gpu_norm"] + df["cpu_norm"] + 1e-9)
        )

        self.log.info(f"Target range  : {df['bottleneck_percent'].min():.1f}  to  {df['bottleneck_percent'].max():.1f}")
        self.log.info(f"Target mean   : {df['bottleneck_percent'].mean():.2f}")
        return df

    def _build_features(self, df: pd.DataFrame):
        """
        Assembles the feature matrix X and target vector y.
        Features: cpuMark, G3Dmark, ratio, cores
        """
        df["ratio"] = df["gpu_norm"] / (df["cpu_norm"] + 1e-9)

        X = df[Config.FEATURE_NAMES].values.astype(np.float64)
        y = df["bottleneck_percent"].values.astype(np.float64)

        self.log.info(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
        return X, y

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        """
        Z-score normalisation:  z = (x - mean) / std

        WHY this matters:
        - cpuMark  range ~100 to 108,000
        - cores    range ~1  to 64
        Without scaling, gradient descent treats large-valued
        features as more important. Standardising levels the field.
        """
        self.X_mean = X.mean(axis=0)
        self.X_std  = X.std(axis=0)
        X_scaled    = (X - self.X_mean) / self.X_std

        for name, m, s in zip(Config.FEATURE_NAMES, self.X_mean, self.X_std):
            self.log.info(f"  {name:<12}  mean={m:.2f}   std={s:.2f}")

        return X_scaled

    def _split(self, X: np.ndarray, y: np.ndarray):
        """
        Manual 80/20 train/test split using index shuffling.
        No sklearn — just numpy permutation.
        """
        np.random.seed(Config.RANDOM_SEED)
        idx   = np.random.permutation(len(X))
        cut   = int((1 - Config.TEST_SIZE) * len(idx))

        self.X_train = X[idx[:cut]]
        self.X_test  = X[idx[cut:]]
        self.y_train = y[idx[:cut]]
        self.y_test  = y[idx[cut:]]

        self.log.success(f"Train : {len(self.X_train):,}  |  Test : {len(self.X_test):,}")

    # ─────────────────────────────────────────────────────
    # Getters — keeps internal arrays private
    # ─────────────────────────────────────────────────────
    def get_train(self):
        return self.X_train, self.y_train

    def get_test(self):
        return self.X_test, self.y_test

    def get_scale_params(self):
        return self.X_mean, self.X_std

    def get_norm_stats(self):
        """Returns the min/max values used to normalise benchmark scores."""
        return {
            "cpu_min": self._cpu_min,
            "cpu_max": self._cpu_max,
            "gpu_min": self._gpu_min,
            "gpu_max": self._gpu_max,
        }

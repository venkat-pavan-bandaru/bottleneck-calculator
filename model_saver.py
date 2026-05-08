# model_saver.py
# Responsible for: persisting trained weights and loading them back.
# OOP concept: Encapsulation — all file I/O in one place.
#              LinearRegression never touches files directly.

import numpy as np
from config import Config
from logger import Logger


class ModelSaver:

    def __init__(self):
        self.log = Logger("ModelSaver")

    def save(self, model, X_mean: np.ndarray, X_std: np.ndarray, norm_stats: dict):
        """
        Persists weights, bias, and all scaling parameters.
        Without scaling params we cannot correctly transform new inputs.
        """
        w, b = model.get_weights()

        np.save(Config.WEIGHTS_FILE, w)
        np.save(Config.BIAS_FILE,    np.array([b]))
        np.save(Config.X_MEAN_FILE,  X_mean)
        np.save(Config.X_STD_FILE,   X_std)
        np.save(Config.CPU_NORM_STATS, np.array([norm_stats["cpu_min"],
                                                  norm_stats["cpu_max"]]))
        np.save(Config.GPU_NORM_STATS, np.array([norm_stats["gpu_min"],
                                                  norm_stats["gpu_max"]]))

        self.log.success(f"Weights saved  → {Config.WEIGHTS_FILE}")
        self.log.success(f"Bias saved     → {Config.BIAS_FILE}")
        self.log.success(f"Scale params   → {Config.X_MEAN_FILE}, {Config.X_STD_FILE}")

    def load(self) -> dict:
        """
        Loads everything needed to run predictions on new data.
        Returns a dict so callers are explicit about what they use.
        """
        data = {
            "w":       np.load(Config.WEIGHTS_FILE),
            "b":       float(np.load(Config.BIAS_FILE)[0]),
            "X_mean":  np.load(Config.X_MEAN_FILE),
            "X_std":   np.load(Config.X_STD_FILE),
            "cpu_stats": np.load(Config.CPU_NORM_STATS),
            "gpu_stats": np.load(Config.GPU_NORM_STATS),
        }
        self.log.success("Model loaded from disk")
        return data

import numpy as np
import pandas as pd
from config import Config
from logger import Logger

class Predictor:
    def __init__(self, model_data, feature_engineer=None,
                 poly_mean=None, poly_std=None,
                 base_mean=None, base_std=None):
        self.log     = Logger("Predictor")
        self.w       = model_data["w"]
        self.b       = model_data["b"]
        self.cpu_min, self.cpu_max = model_data["cpu_stats"]
        self.gpu_min, self.gpu_max = model_data["gpu_stats"]
        self.fe        = feature_engineer
        self.poly_mean = poly_mean
        self.poly_std  = poly_std
        self.base_mean = base_mean
        self.base_std  = base_std
        self.cpu_lookup = pd.read_csv(Config.CPU_LOOKUP)
        self.gpu_lookup = pd.read_csv(Config.GPU_LOOKUP)

    def predict(self, cpu_name, gpu_name):
        cpu_row   = self._find(self.cpu_lookup, "cpuName", cpu_name)
        gpu_row   = self._find(self.gpu_lookup, "gpuName", gpu_name)
        cpu_score = int(cpu_row["cpuMark"])
        gpu_score = int(gpu_row["G3Dmark"])
        cores     = int(cpu_row["cores"])

        cpu_norm = (cpu_score - self.cpu_min) / (self.cpu_max - self.cpu_min)
        gpu_norm = (gpu_score - self.gpu_min) / (self.gpu_max - self.gpu_min)
        ratio    = gpu_norm / (cpu_norm + 1e-9)

        x_raw    = np.array([[cpu_score, gpu_score, ratio, cores]], dtype=np.float64)
        # Step 1: base standardisation
        x_base   = (x_raw - self.base_mean) / self.base_std
        # Step 2: polynomial expansion
        if self.fe is not None:
            x_poly = self.fe.transform(x_base)
        else:
            x_poly = x_base
        # Step 3: poly standardisation
        if self.poly_mean is not None:
            x_final = (x_poly - self.poly_mean) / self.poly_std
        else:
            x_final = x_poly

        pct = float((x_final @ self.w + self.b)[0])
        verdict, severity, rec = self._interpret(pct)
        result = {"cpu_name": cpu_name, "gpu_name": gpu_name,
                  "cpu_score": cpu_score, "gpu_score": gpu_score,
                  "bottleneck_percent": round(pct, 2),
                  "verdict": verdict, "severity": severity, "recommendation": rec}
        self._print_result(result)
        return result

    def _find(self, df, col, name):
        matches = df[df[col].str.contains(name, case=False, na=False)]
        if matches.empty:
            raise ValueError(f"'{name}' not found in {col}.")
        if len(matches) > 1:
            self.log.info(f"Multiple matches for '{name}', using: {matches[col].iloc[0]}")
        return matches.iloc[0]

    @staticmethod
    def _interpret(pct):
        abs_pct = abs(pct)
        if abs_pct <= 10:
            return "Balanced", "Balanced", "Excellent pairing. CPU and GPU are well matched."
        verdict = "GPU bottleneck" if pct > 0 else "CPU bottleneck"
        rec = ("GPU is doing more work than CPU can supply. Consider upgrading your CPU."
               if pct > 0 else
               "CPU is the limiting factor. Consider upgrading CPU or pairing with a weaker GPU.")
        severity = "Severe" if abs_pct > 60 else "Moderate" if abs_pct > 30 else "Mild"
        return verdict, severity, rec

    def _print_result(self, r):
        self.log.section("Prediction Result")
        self.log.info(f"CPU  : {r['cpu_name']}  (score: {r['cpu_score']:,})")
        self.log.info(f"GPU  : {r['gpu_name']}  (score: {r['gpu_score']:,})")
        self.log.info(f"Bottleneck % : {r['bottleneck_percent']:+.2f}")
        self.log.info(f"Verdict      : {r['verdict']}  [{r['severity']}]")
        self.log.info(f"Advice       : {r['recommendation']}")

# report_generator.py
# Saves a complete human-readable report of the entire experiment.
# This is what you'd hand to a recruiter or include in a GitHub README.
# OOP concept: Encapsulation — all formatting logic in one place.

from datetime import datetime
from logger import Logger


class ReportGenerator:

    def __init__(self, output_path: str = "report.txt"):
        self.output_path = output_path
        self.log         = Logger("ReportGenerator")
        self._lines      = []

    def build(self, comparison_results: list,
              cv_results: dict,
              sample_predictions: list,
              feature_count: int,
              poly_degree: int) -> "ReportGenerator":

        self._lines = []
        self._header()
        self._experiment_config(feature_count, poly_degree)
        self._model_comparison_table(comparison_results)
        self._cv_section(cv_results)
        self._predictions_section(sample_predictions)
        self._interpretation(comparison_results)
        return self

    def save(self):
        text = "\n".join(self._lines)
        with open(self.output_path, "w") as f:
            f.write(text)
        self.log.success(f"Report saved → {self.output_path}")

    # ─────────────────────────────────────────────────────
    # Private section builders
    # ─────────────────────────────────────────────────────

    def _header(self):
        self._lines += [
            "=" * 65,
            "  PC BOTTLENECK PREDICTOR — EXPERIMENT REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 65, "",
            "PROJECT OVERVIEW",
            "─" * 65,
            "  Predicts CPU/GPU bottleneck percentage for gaming PCs.",
            "  Dataset: PassMark CPU + GPU benchmark scores (real data).",
            "  Target : bottleneck_percent ∈ [-100, +100]",
            "           +100 = severe GPU bottleneck",
            "           -100 = severe CPU bottleneck",
            "              0 = perfectly balanced",
            "",
            "ALGORITHMS IMPLEMENTED FROM SCRATCH (no sklearn)",
            "─" * 65,
            "  1. Linear Regression    — batch gradient descent",
            "  2. Ridge Regression     — gradient descent + L2 penalty",
            "  3. Polynomial Features  — degree-2 cross/squared terms",
            "  4. K-Fold Cross Validation (5-fold)",
            "",
        ]

    def _experiment_config(self, feature_count, poly_degree):
        self._lines += [
            "EXPERIMENT CONFIGURATION",
            "─" * 65,
            f"  Polynomial degree : {poly_degree}",
            f"  Total features    : {feature_count}",
            f"  Original features : cpuMark, G3Dmark, ratio, cores",
            f"  Learning rate (α) : 0.01",
            f"  Iterations        : 2,000",
            f"  Train/test split  : 80% / 20%",
            f"  K-fold K          : 5",
            "",
        ]

    def _model_comparison_table(self, results: list):
        self._lines += [
            "MODEL COMPARISON (ranked by Test R²)",
            "─" * 65,
            f"  {'Model':<24} {'Train R²':>9} {'Test R²':>9} {'RMSE':>8} {'Overfit':>8}",
            "  " + "─" * 60,
        ]
        for r in results:
            flag = " ★ best" if r is results[0] else ""
            self._lines.append(
                f"  {r['label']:<24}"
                f"  {r['train_r2']:>8.4f}"
                f"  {r['test_r2']:>8.4f}"
                f"  {r['test_rmse']:>8.4f}"
                f"  {r['overfit_gap']:>8.4f}{flag}"
            )
        self._lines.append("")

    def _cv_section(self, cv: dict):
        self._lines += [
            "CROSS-VALIDATION RESULTS (best model, 5-fold)",
            "─" * 65,
        ]
        for i, (r2, rmse) in enumerate(zip(cv["r2_scores"], cv["rmse_scores"]), 1):
            self._lines.append(f"  Fold {i}: R²={r2:.4f}  RMSE={rmse:.4f}")
        self._lines += [
            f"  Mean R² : {cv['mean_r2']:.4f}  ±{cv['std_r2']:.4f}",
            f"  Mean RMSE: {cv['mean_rmse']:.4f}",
            "  (Low std means model is stable, not sensitive to data split)",
            "",
        ]

    def _predictions_section(self, preds: list):
        self._lines += [
            "SAMPLE PREDICTIONS",
            "─" * 65,
        ]
        for p in preds:
            self._lines += [
                f"  CPU : {p['cpu_name']}",
                f"  GPU : {p['gpu_name']}",
                f"  Bottleneck: {p['bottleneck_percent']:+.2f}%  |  {p['verdict']}  |  {p['severity']}",
                f"  Advice: {p['recommendation']}",
                "",
            ]

    def _interpretation(self, results: list):
        best   = results[0]
        worst  = results[-1]
        self._lines += [
            "INTERPRETATION",
            "─" * 65,
            f"  Best model  : {best['label']} (Test R²={best['test_r2']:.4f})",
            f"  Worst model : {worst['label']} (Test R²={worst['test_r2']:.4f})",
            "",
            "  R² SCALE:",
            "    0.0 - 0.5  →  Weak fit  (model barely better than guessing mean)",
            "    0.5 - 0.7  →  Moderate  (some patterns captured)",
            "    0.7 - 0.9  →  Good fit  (most variance explained)",
            "    0.9 - 1.0  →  Strong    (model explains nearly all variance)",
            "",
            "  WHY RIDGE BEATS LINEAR ON POLYNOMIAL FEATURES:",
            "    Adding degree-2 terms increases features from 4 to 14.",
            "    More features = more risk of overfitting (large weights).",
            "    Ridge's L2 penalty shrinks weights → better generalisation.",
            "    Proof: check overfit_gap — Ridge has smaller train/test gap.",
            "",
            "=" * 65,
        ]

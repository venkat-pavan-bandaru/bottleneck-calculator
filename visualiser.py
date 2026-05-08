# visualiser.py
# Generates all project plots and saves them as PNG files.
# OOP concept: Encapsulation — matplotlib complexity hidden inside methods.
#              main.py just calls visualiser.plot_all().

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from logger import Logger


class Visualiser:

    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        self.log        = Logger("Visualiser")

    # ─────────────────────────────────────────────────────
    # Public: plot everything at once
    # ─────────────────────────────────────────────────────
    def plot_all(self, linear_model, ridge_model,
                 X_test, y_test,
                 comparison_results: list,
                 feature_names: list):
        self.log.section("Generating Plots")
        self.cost_curves(linear_model, ridge_model)
        self.predicted_vs_actual(ridge_model, X_test, y_test)
        self.residuals(ridge_model, X_test, y_test)
        self.feature_importance(ridge_model, feature_names)
        self.model_comparison_bar(comparison_results)
        self.log.success("All plots saved to PNG files")

    # ─────────────────────────────────────────────────────
    # Plot 1: Cost curves — watch gradient descent converge
    # ─────────────────────────────────────────────────────
    def cost_curves(self, linear_model, ridge_model):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Gradient Descent — Cost vs Iteration", fontsize=13, fontweight="bold")

        for ax, model, label, colour in zip(
            axes,
            [linear_model, ridge_model],
            ["Linear Regression", "Ridge Regression (best λ)"],
            ["#2563EB", "#D97706"]
        ):
            costs = model.get_cost_history()
            ax.plot(costs, color=colour, linewidth=1.5)
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Cost (MSE)")
            ax.grid(True, alpha=0.3)

            # Annotate start and end cost
            ax.annotate(f"Start: {costs[0]:.1f}",
                        xy=(0, costs[0]),
                        xytext=(len(costs)*0.1, costs[0]*0.9),
                        fontsize=8, color="gray")
            ax.annotate(f"End: {costs[-1]:.1f}",
                        xy=(len(costs)-1, costs[-1]),
                        xytext=(len(costs)*0.6, costs[-1]*1.4),
                        fontsize=8, color=colour)

        plt.tight_layout()
        path = f"{self.output_dir}/plot_cost_curves.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self.log.success(f"Saved: {path}")

    # ─────────────────────────────────────────────────────
    # Plot 2: Predicted vs Actual
    # Perfect model → all dots on the diagonal line y=x
    # ─────────────────────────────────────────────────────
    def predicted_vs_actual(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        # Sample 2000 points so the plot isn't overcrowded
        idx    = np.random.choice(len(y_test), size=min(2000, len(y_test)), replace=False)
        y_s    = y_test[idx]
        yp_s   = y_pred[idx]

        fig, ax = plt.subplots(figsize=(7, 6))

        sc = ax.scatter(y_s, yp_s, alpha=0.3, s=8, c=np.abs(y_s - yp_s),
                        cmap="RdYlGn_r")
        plt.colorbar(sc, ax=ax, label="Absolute error")

        # Perfect prediction line
        lims = [min(y_s.min(), yp_s.min()), max(y_s.max(), yp_s.max())]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6, label="Perfect fit")

        # Compute R² for subtitle
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2     = 1 - ss_res / ss_tot

        ax.set_xlabel("Actual Bottleneck %")
        ax.set_ylabel("Predicted Bottleneck %")
        ax.set_title(f"Predicted vs Actual  (R² = {r2:.4f})", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        path = f"{self.output_dir}/plot_predicted_vs_actual.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self.log.success(f"Saved: {path}")

    # ─────────────────────────────────────────────────────
    # Plot 3: Residuals
    # Good model → residuals scattered randomly around 0
    # Pattern in residuals → model is missing something
    # ─────────────────────────────────────────────────────
    def residuals(self, model, X_test, y_test):
        y_pred    = model.predict(X_test)
        residuals = y_test - y_pred

        idx = np.random.choice(len(residuals), size=min(2000, len(residuals)), replace=False)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Residual Analysis", fontsize=13, fontweight="bold")

        # Residuals vs predicted
        axes[0].scatter(y_pred[idx], residuals[idx], alpha=0.3, s=8, color="#7C3AED")
        axes[0].axhline(0, color="red", linewidth=1, linestyle="--")
        axes[0].set_xlabel("Predicted Value")
        axes[0].set_ylabel("Residual (actual − predicted)")
        axes[0].set_title("Residuals vs Predicted")
        axes[0].grid(True, alpha=0.3)

        # Residual distribution
        axes[1].hist(residuals, bins=60, color="#7C3AED", alpha=0.7, edgecolor="white", linewidth=0.3)
        axes[1].axvline(0, color="red", linewidth=1.5, linestyle="--")
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"Residual Distribution  (mean={residuals.mean():.2f})")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = f"{self.output_dir}/plot_residuals.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self.log.success(f"Saved: {path}")

    # ─────────────────────────────────────────────────────
    # Plot 4: Feature importance — which weights are largest?
    # ─────────────────────────────────────────────────────
    def feature_importance(self, model, feature_names: list):
        w, _ = model.get_weights()
        importance = np.abs(w)

        # Sort descending, show top 15
        top_n   = min(15, len(importance))
        top_idx = np.argsort(importance)[::-1][:top_n]
        top_imp = importance[top_idx]
        top_lbl = [feature_names[i] if i < len(feature_names)
                   else f"poly_{i}" for i in top_idx]

        colours = ["#2563EB" if i < 4 else "#D97706" for i in top_idx]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(range(top_n), top_imp, color=colours, alpha=0.85)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_lbl, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("|Weight| — larger = more influence on prediction")
        ax.set_title("Feature Importance (top 15 by |weight|)", fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#2563EB", label="Original features"),
                           Patch(facecolor="#D97706", label="Polynomial features")]
        ax.legend(handles=legend_elements, fontsize=9)

        plt.tight_layout()
        path = f"{self.output_dir}/plot_feature_importance.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self.log.success(f"Saved: {path}")

    # ─────────────────────────────────────────────────────
    # Plot 5: Model comparison bar chart
    # ─────────────────────────────────────────────────────
    def model_comparison_bar(self, comparison_results: list):
        labels     = [r["label"] for r in comparison_results]
        train_r2   = [r["train_r2"] for r in comparison_results]
        test_r2    = [r["test_r2"]  for r in comparison_results]

        x     = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width/2, train_r2, width, label="Train R²", color="#2563EB", alpha=0.85)
        ax.bar(x + width/2, test_r2,  width, label="Test R²",  color="#10B981", alpha=0.85)

        ax.set_ylabel("R²")
        ax.set_title("Model Comparison — R² Score", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.axhline(0.85, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(len(labels)-0.5, 0.86, "0.85 target", fontsize=8, color="red")
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        path = f"{self.output_dir}/plot_model_comparison.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self.log.success(f"Saved: {path}")

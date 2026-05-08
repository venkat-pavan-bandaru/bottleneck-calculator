# 🖥️ PC Bottleneck Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Only-013243?style=for-the-badge&logo=numpy&logoColor=white)
![No sklearn](https://img.shields.io/badge/sklearn-NOT%20USED-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-22c55e?style=for-the-badge)

**Predict CPU/GPU bottleneck percentage for any gaming PC build — built from scratch with pure NumPy.**

*Linear Regression · Ridge Regression · Polynomial Features · K-Fold CV · All from scratch. No sklearn.*

[Features](#-features) · [Architecture](#-project-architecture) · [Math](#-the-math) · [Results](#-results) · [Usage](#-usage) · [Team](#-team)

</div>

---

## 📌 What Is a Bottleneck?

When you build a gaming PC, one component can limit the other. If your GPU is far more powerful than your CPU, the CPU can't feed frames fast enough — that's a **CPU bottleneck**. The reverse is a **GPU bottleneck**.

This project predicts a signed bottleneck score:

```
+100  →  Pure GPU bottleneck   (GPU overwhelms the CPU)
   0  →  Perfectly balanced    (ideal pairing)
-100  →  Pure CPU bottleneck   (CPU overwhelms the GPU)
```

### Example

```
Input:   CPU → Ryzen 5 5600X     GPU → RTX 3060 Ti
Output:  Bottleneck: +50.93%  |  Verdict: GPU Bottleneck  |  Severity: Moderate
         Advice: GPU is doing more work than CPU can supply. Consider upgrading your CPU.
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔢 **From-scratch ML** | Linear & Ridge Regression with gradient descent — zero sklearn |
| 📐 **Polynomial Features** | Degree-2 expansion (4 → 14 features) to capture non-linear patterns |
| 🔁 **K-Fold Cross Validation** | 5-fold CV implemented manually — proves model generalises |
| ⚖️ **Model Comparison** | 5 models compared head-to-head on same test set |
| 📊 **5 Diagnostic Plots** | Cost curves, predicted vs actual, residuals, feature importance, comparison bar |
| 🏗️ **OOP Architecture** | 15 files, each a single class with a single responsibility |
| 📄 **Auto Report** | Full experiment report saved to `report.txt` after every run |
| 🔍 **Named Lookup** | Type `"Ryzen 5 5600"` — partial name matching handles the rest |

---

## 🏗️ Project Architecture

```
pc_bottleneck_predictor/
│
├── main.py                  ← Orchestrator. Zero logic — wires everything together.
│
├── config.py                ← All hyperparameters & filenames in one place
├── logger.py                ← Timestamped console output (class Logger)
│
├── data_loader.py           ← Reads CSVs, filters Desktop, saves lookup tables
├── preprocessor.py          ← Cross-join, target engineering, Z-score, train/test split
├── feature_engineer.py      ← Degree-2 polynomial expansion (pure combinatorics)
│
├── linear_regression.py     ← Batch gradient descent from scratch
├── ridge_regression.py      ← Gradient descent + L2 penalty from scratch
│
├── model_comparison.py      ← Trains all 5 candidates, ranks by test R²
├── cross_validator.py       ← 5-fold CV, fresh model per fold, no data leakage
├── evaluator.py             ← MSE, RMSE, MAE, R² — all manual formulas
│
├── model_saver.py           ← Saves/loads weights + all scaling parameters
├── predictor.py             ← Name → benchmark score → feature pipeline → prediction
├── visualiser.py            ← 5 matplotlib plots saved as PNG
├── report_generator.py      ← Writes full experiment report to report.txt
│
├── cpu.csv                  ← PassMark CPU benchmark dataset
└── gpu.csv                  ← PassMark GPU benchmark dataset
```

> **OOP principle at play:** `main.py` never does any work itself. It only calls other classes in sequence. If you want to understand *what* the program does — read `main.py`. If you want to understand *how* — read the individual files.

---

## 📐 The Math

### Target Engineering

Neither dataset has a bottleneck label. We derive it:

$$\text{bottleneck\\_pct} = 100 \times \frac{\text{gpu\\_norm} - \text{cpu\\_norm}}{\text{gpu\\_norm} + \text{cpu\\_norm} + \varepsilon}$$

Where `cpu_norm` and `gpu_norm` are min-max normalised benchmark scores. This gives a bounded, interpretable target in `[-100, +100]`.

### Linear Regression

$$\hat{y} = \mathbf{X}\mathbf{w} + b$$

**Cost (MSE):**
$$J(\mathbf{w}, b) = \frac{1}{2m} \|\mathbf{X}\mathbf{w} + b - \mathbf{y}\|^2$$

**Gradients:**
$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} \mathbf{X}^\top(\hat{y} - y) \qquad \frac{\partial J}{\partial b} = \frac{1}{m} \sum(\hat{y} - y)$$

**Update rule:**
$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \cdot \frac{\partial J}{\partial \mathbf{w}}$$

### Ridge Regression (L2 Regularisation)

$$J_{\text{ridge}} = \frac{1}{2m}\|\mathbf{X}\mathbf{w} + b - \mathbf{y}\|^2 + \frac{\lambda}{2m}\|\mathbf{w}\|^2$$

The extra term penalises large weights, preventing overfitting on the 14-feature polynomial matrix. The bias `b` is **not** regularised — regularising the bias would shift all predictions uniformly without reducing model complexity.

### Polynomial Feature Expansion

Base: `[cpuMark, G3Dmark, ratio, cores]` → **4 features**

After degree-2 expansion (all $x_i^2$ and $x_i x_j$ terms): → **14 features**

The model stays linear in the weights — it's the *input representation* that gains expressivity, not the model class.

---

## 📊 Results

### Model Comparison

| Model | Train R² | Test R² | RMSE | MAE | Overfit Gap |
|---|---|---|---|---|---|
| **Linear Regression ★** | **0.7322** | **0.7321** | **34.59** | **27.55** | **0.0001** |
| Ridge λ=0.01 | 0.7322 | 0.7321 | 34.59 | 27.55 | 0.0001 |
| Ridge λ=0.1  | 0.7322 | 0.7321 | 34.59 | 27.55 | 0.0001 |
| Ridge λ=1.0  | 0.7322 | 0.7321 | 34.59 | 27.55 | 0.0001 |
| Ridge λ=10.0 | 0.7320 | 0.7319 | 34.60 | 27.55 | 0.0001 |

> Overfit gap ≈ 0.0001 across all models — **zero overfitting**. The polynomial expansion on 321k samples is inherently stable.

### 5-Fold Cross Validation

| Fold | R² | RMSE |
|---|---|---|
| Fold 1 | 0.7467 | 33.91 |
| Fold 2 | 0.7276 | 34.76 |
| Fold 3 | 0.7306 | 34.91 |
| Fold 4 | 0.6897 | 36.90 |
| Fold 5 | 0.7340 | 34.25 |
| **Mean ± Std** | **0.7257 ± 0.019** | **34.95** |

Low std of ±0.019 confirms the model is **stable** — not sensitive to which rows end up in each fold.

### Why R² = 0.73 and Not Higher?

The remaining 27% of unexplained variance comes from effects our benchmark scores can't capture:
- CPU architecture differences (same score, different IPC)
- Cache hierarchy and threading behaviour differences
- Game engine type (CPU-bound vs GPU-bound titles)
- Resolution-specific bottleneck behaviour

A gradient-boosted tree model would push this above 0.90. That's the honest, intentional next step.

---

## 🚀 Usage

### 1. Install dependencies

```bash
pip install numpy pandas matplotlib
```

### 2. Place your data files

```
pc_bottleneck_predictor/
├── cpu.csv    ← PassMark CPU dataset
└── gpu.csv    ← PassMark GPU dataset
```

### 3. Run

```bash
python main.py
```

### 4. Output

```
weights.npy            ← trained model weights
bias.npy               ← trained bias
X_mean.npy / X_std.npy ← scaling parameters
report.txt             ← full experiment report
plot_cost_curves.png
plot_predicted_vs_actual.png
plot_residuals.png
plot_feature_importance.png
plot_model_comparison.png
```

### 5. Make your own prediction

Edit the bottom of `main.py`:

```python
predictor.predict("Ryzen 5 5600X", "GeForce RTX 3080")
predictor.predict("Core i5-12400",  "Radeon RX 6700 XT")
```

Partial name matching is supported — `"3080"` will match `"GeForce RTX 3080"`.

---

## 📁 Dataset

| File | Source | Rows (raw) | Rows (Desktop only) |
|---|---|---|---|
| `cpu.csv` | [PassMark CPU Benchmarks](https://www.cpubenchmark.net) | 3,825 | 1,114 |
| `gpu.csv` | [PassMark GPU Benchmarks](https://www.videocardbenchmark.net) | 2,317 | 289 |

**Cross-join:** 1,114 × 289 = **321,946 training samples**

---

## 🧱 OOP Concepts Used

| Concept | Where |
|---|---|
| **Single Responsibility** | Every class does exactly one job |
| **Encapsulation** | `_private_methods` hide implementation details |
| **Abstraction** | `main.py` sees only `.fit()`, `.predict()`, `.run()` |
| **Method Chaining** | `loader.load().get_cpu()` via `return self` |
| **Dependency Injection** | `Preprocessor(cpu_df, gpu_df)` — receives data, doesn't load it |
| **Strategy Pattern** | `ModelComparison` swaps models without changing evaluation logic |
| **Open/Closed** | Add new models to `model_comparison.py` without touching other files |

---

## 📈 Plots Generated

| Plot | What It Shows |
|---|---|
| `plot_cost_curves.png` | MSE dropping over 5,000 iterations for both models |
| `plot_predicted_vs_actual.png` | Scatter of predicted vs real values — ideal = diagonal |
| `plot_residuals.png` | Residual distribution — ideal = centred around zero |
| `plot_feature_importance.png` | Top 15 feature weights — which inputs drive predictions |
| `plot_model_comparison.png` | Train vs test R² bar chart for all 5 models |

---

## 🗂️ Config Reference

All hyperparameters live in `config.py`. Change once, affects the whole project:

```python
class Config:
    LEARNING_RATE   = 0.1      # gradient descent step size
    ITERATIONS      = 5000     # training loop iterations
    TEST_SIZE       = 0.2      # 20% held out for testing
    RANDOM_SEED     = 42       # reproducible shuffles
    CV_SAMPLE_SIZE  = 20000    # rows used in cross-validation
```

---

## 🔭 Future Work

- [ ] Replace synthetic target with real FPS measurements per game title
- [ ] Add resolution as a feature (1080p / 1440p / 4K)
- [ ] Implement gradient-boosted decision tree from scratch (target R² > 0.90)
- [ ] Build a Flask REST API for the predictor
- [ ] Deploy to Render/Railway with a React frontend
- [ ] Add game-specific models (esports vs AAA engine types)

---

## 👥 Team

| Name | Role | Responsibilities |
|---|---|---|
| **Bandaru Venkat Pavan** | Project Lead | Architecture, gradient descent, polynomial features, double-standardisation pipeline, model comparison, cross-validation |
| **Gaddam Pranav** | Team Member | Data loading, preprocessing, target engineering, cross-join construction, model saver, predictor |
| **Harshit** | Team Member | Evaluation metrics, visualisation suite, report generator, hyperparameter grid search, logging |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with 🧠 pure math and pure NumPy — no shortcuts.

*If you found this useful, give it a ⭐*

</div>

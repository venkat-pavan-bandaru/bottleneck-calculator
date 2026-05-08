# main.py — orchestrator
import numpy as np
from data_loader        import DataLoader
from preprocessor       import Preprocessor
from feature_engineer   import FeatureEngineer
from linear_regression  import LinearRegression
from ridge_regression   import RidgeRegression
from model_comparison   import ModelComparison
from cross_validator    import CrossValidator
from evaluator          import Evaluator
from model_saver        import ModelSaver
from predictor          import Predictor
from visualiser         import Visualiser
from report_generator   import ReportGenerator
from logger             import Logger
from config             import Config

log = Logger("Main")

# ── 1. Load ───────────────────────────────────────────
loader = DataLoader()
loader.load()

# ── 2. Preprocess ─────────────────────────────────────
prep = Preprocessor(loader.get_cpu(), loader.get_gpu())
prep.run()
X_train_base, y_train = prep.get_train()
X_test_base,  y_test  = prep.get_test()
X_mean_4, X_std_4     = prep.get_scale_params()
norm_stats            = prep.get_norm_stats()

# ── 3. Polynomial expansion + second standardisation ──
fe = FeatureEngineer(degree=2)
X_train_poly = fe.fit_transform(X_train_base)
X_test_poly  = fe.transform(X_test_base)

X_poly_mean = X_train_poly.mean(axis=0)
X_poly_std  = X_train_poly.std(axis=0) + 1e-8
X_train     = (X_train_poly - X_poly_mean) / X_poly_std
X_test      = (X_test_poly  - X_poly_mean) / X_poly_std

all_feature_names = Config.FEATURE_NAMES + [
    f"poly_{'x'.join(map(str,c))}" for c in fe.combinations
]
log.info(f"Final feature matrix: {X_train.shape}")

# ── 4. Model comparison (on 30k sample for speed) ─────
#    WHY sample? 321k rows × 5 models × 5000 iters is slow.
#    30k rows is statistically representative — same patterns,
#    fraction of the compute. Winner retrained on full data below.
np.random.seed(Config.RANDOM_SEED)
compare_sample_size = min(len(X_train), Config.MODEL_COMPARE_SAMPLE_SIZE)
sample_idx  = np.random.choice(len(X_train), size=compare_sample_size, replace=False)
X_samp      = X_train[sample_idx]
y_samp      = y_train[sample_idx]

comparison = ModelComparison()
comparison.run(X_samp, y_samp, X_test, y_test)
best_label = comparison.get_results()[0]["label"]

# ── 5. Retrain winner on FULL training set ─────────────
log.section(f"Retraining winner on full data: {best_label}")
if "Linear" in best_label:
    final_model = LinearRegression()
    final_model.fit(X_train, y_train, lr=Config.LEARNING_RATE,
                    iterations=Config.ITERATIONS)
else:
    lam = float(best_label.split("=")[1])
    final_model = RidgeRegression(lambda_=lam)
    final_model.fit(X_train, y_train, lr=Config.LEARNING_RATE,
                    iterations=Config.ITERATIONS)

# ── 6. Cross validation (20k sample) ──────────────────
X_full     = np.vstack([X_train, X_test])
y_full     = np.concatenate([y_train, y_test])
cv_sample_size = min(len(X_full), Config.CV_SAMPLE_SIZE)
cv_idx     = np.random.choice(len(X_full), size=cv_sample_size, replace=False)
X_cv, y_cv = X_full[cv_idx], y_full[cv_idx]

if "Linear" in best_label:
    cv_class, cv_kwargs = LinearRegression, {}
else:
    cv_class, cv_kwargs = RidgeRegression, {"lambda_": lam}

cv = CrossValidator(k=5)
cv_results = cv.validate(cv_class, X_cv, y_cv,
                          model_kwargs=cv_kwargs, name=best_label)

# ── 7. Final evaluation ────────────────────────────────
evaluator = Evaluator()
evaluator.evaluate(y_test, final_model.predict(X_test), label="Final Model — Test")

# ── 8. Save ───────────────────────────────────────────
saver = ModelSaver()
saver.save(final_model, X_poly_mean, X_poly_std, norm_stats)

# ── 9. Predictions ─────────────────────────────────────
model_data = saver.load()
predictor  = Predictor(model_data, feature_engineer=fe,
                       poly_mean=X_poly_mean, poly_std=X_poly_std,
                       base_mean=X_mean_4,    base_std=X_std_4)

sample_preds = [
    predictor.predict("Ryzen 5 5600",  "GeForce RTX 3060"),
    predictor.predict("Ryzen 9 5900X", "Radeon RX 6600"),
    predictor.predict("Ryzen 9 5900X", "GeForce RTX 3080"),
]

# ── 10. Visualisations ────────────────────────────────
linear_ref = next(r["model"] for r in comparison.get_results()
                  if "Linear" in r["label"])
vis = Visualiser(output_dir=".")
vis.plot_all(linear_ref, final_model, X_test, y_test,
             comparison.get_results(), all_feature_names)

# ── 11. Report ────────────────────────────────────────
ReportGenerator("report.txt").build(
    comparison_results = comparison.get_results(),
    cv_results         = cv_results,
    sample_predictions = sample_preds,
    feature_count      = X_train.shape[1],
    poly_degree        = 2,
).save()

log.section("Complete")
log.success("report.txt  |  plot_*.png  |  weights.npy")

from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent

    CPU_CSV      = BASE_DIR / "cpu.csv"
    GPU_CSV      = BASE_DIR / "gpu.csv"
    CPU_LOOKUP   = BASE_DIR / "cpu_lookup.csv"
    GPU_LOOKUP   = BASE_DIR / "gpu_lookup.csv"

    # Proven hyperparameters from grid search
    LEARNING_RATE   = 0.1
    ITERATIONS      = 5000
    TEST_SIZE       = 0.2
    RANDOM_SEED     = 42

    FEATURE_NAMES = ["cpuMark", "G3Dmark", "ratio", "cores"]

    WEIGHTS_FILE   = BASE_DIR / "weights.npy"
    BIAS_FILE      = BASE_DIR / "bias.npy"
    X_MEAN_FILE    = BASE_DIR / "X_mean.npy"
    X_STD_FILE     = BASE_DIR / "X_std.npy"
    CPU_NORM_STATS = BASE_DIR / "cpu_norm_stats.npy"
    GPU_NORM_STATS = BASE_DIR / "gpu_norm_stats.npy"

    LOG_EVERY_N_ITERS = 500
    CV_SAMPLE_SIZE    = 20000
    MODEL_COMPARE_SAMPLE_SIZE = 30000

# data_loader.py
# Responsible for: reading CSVs, filtering, returning clean DataFrames.
# OOP concept: Single Responsibility — this class only knows about loading data.
#              Nothing else in the project reads CSV files directly.

import pandas as pd
from pathlib import Path
from config import Config
from logger import Logger


class DataLoader:

    def __init__(self):
        self.log = Logger("DataLoader")
        self.cpu_df = None
        self.gpu_df = None

    def load(self):
        """
        Reads both CSVs, filters to Desktop category,
        keeps only the columns the model needs.
        Returns self so calls can be chained.
        """
        self.log.section("Loading Data")

        # ── Load raw files ────────────────────────────────
        missing = []
        for csv_path in (Config.CPU_CSV, Config.GPU_CSV):
            if not Path(csv_path).exists():
                missing.append(str(csv_path))

        if missing:
            raise FileNotFoundError(
                "Missing required dataset file(s): "
                + ", ".join(missing)
                + f". Current working directory: {Path.cwd()}"
            )

        cpu_raw = pd.read_csv(Config.CPU_CSV)
        gpu_raw = pd.read_csv(Config.GPU_CSV)

        self.log.info(f"CPU raw rows : {len(cpu_raw):,}")
        self.log.info(f"GPU raw rows : {len(gpu_raw):,}")

        # ── Filter: Desktop only, drop nulls ─────────────
        self.cpu_df = (
            cpu_raw[cpu_raw["category"] == "Desktop"]
            [["cpuName", "cpuMark", "cores", "TDP"]]
            .dropna()
            .rename(columns={"TDP": "cpu_TDP"})
            .reset_index(drop=True)
        )

        self.gpu_df = (
            gpu_raw[gpu_raw["category"] == "Desktop"]
            [["gpuName", "G3Dmark", "TDP"]]
            .dropna()
            .rename(columns={"TDP": "gpu_TDP"})
            .reset_index(drop=True)
        )

        self.log.success(f"Desktop CPUs : {len(self.cpu_df):,}")
        self.log.success(f"Desktop GPUs : {len(self.gpu_df):,}")

        # ── Save lookup tables for use at prediction time ─
        self.cpu_df[["cpuName", "cpuMark", "cores"]].to_csv(Config.CPU_LOOKUP, index=False)
        self.gpu_df[["gpuName", "G3Dmark"]].to_csv(Config.GPU_LOOKUP, index=False)

        self.log.success("Lookup tables saved")

        return self   # allows  loader.load().get_cpu()

    def get_cpu(self) -> pd.DataFrame:
        return self.cpu_df

    def get_gpu(self) -> pd.DataFrame:
        return self.gpu_df

"""
data_pipeline.py

Cleaning, feature engineering, PWM-target construction, splitting, and
train-only augmentation for the BMCDATA dataset, per the project's
methodology (sections 2-4).

    python data_pipeline.py --data-dir ../bmcdata_repo/data --out-dir data/processed

Design notes / things that needed a concrete decision the spec left open:

* Processor temperature (T): averaged from Cpu1_Temp and Cpu2_Temp (this is
  a dual-socket server -- scidata.py, the dataset's own analysis script,
  does the same averaging).
* Workload (W): the dataset's real `cpu_usage` column (host-collected),
  used directly -- no synthetic Light/Medium/Heavy levels needed, unlike
  the earlier physical-acquisition version of this project, since real
  utilization is actually measured here.
* PWM target label: this is the one genuinely hard part, flagged in the
  spec itself (section 6, "PWM Target Generation"). The raw dataset does
  NOT contain a per-row "optimal PWM" column -- FAN1-4 are fan *speeds*
  (RPM), and the applied PWM command is file-level metadata (see
  bmcdata_inventory.csv), not a per-timestep measurement. Only files
  tagged fan_mode_type == FIXED_PWM give a clean, directly-observed label
  (every row in that file really was driven at that exact PWM%). PID/TEAP/
  HyperFan files have no directly observable PWM value in this dataset's
  columns at all. This pipeline:
    - Uses FIXED_PWM rows as direct supervision (pwm_label_source="direct")
    - For everything else, falls back to a heuristic target computed from
      thermal safety + temperature trend + workload + cooling effort
      (pwm_target_heuristic(), pwm_label_source="heuristic") -- a stand-in,
      not a validated optimal-control label. train.py downweights these
      rows relative to direct labels; treat any reported PWM-head accuracy
      as accuracy against "direct" rows specifically if you want a number
      you can defend.
  Both future_temp_c and thermal_risk_score, by contrast, use ALL rows
  regardless of fan mode, since temperature itself is measured either way.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from bmcdata_loader import load_dataset, load_inventory

THERMAL_LIMIT_C = 85.0   # placeholder server CPU thermal ceiling -- tune to your actual platform's spec
SAFE_LOW_C = 30.0
FUTURE_HORIZON_ROWS = 5  # predict ~5 samples ahead (sampling interval varies by file, roughly 5-30s)

FEATURE_COLUMNS = ["proc_temp_c", "dTdt_c_per_s", "inlet_temp_c", "cpu_usage"]
TARGET_COLUMNS = ["pwm_pct", "future_temp_c", "thermal_risk_score"]


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    required = ["Cpu1_Temp", "Cpu2_Temp", "Inlet_Temp", "cpu_usage", "_time"]
    df = df.dropna(subset=required)

    df = df[(df["Cpu1_Temp"] > 0) & (df["Cpu1_Temp"] < 120)]
    df = df[(df["Cpu2_Temp"] > 0) & (df["Cpu2_Temp"] < 120)]
    df = df[(df["Inlet_Temp"] > -10) & (df["Inlet_Temp"] < 60)]
    df = df[(df["cpu_usage"] >= 0) & (df["cpu_usage"] <= 100)]

    df = df.drop_duplicates(subset=["source_file", "_time"])
    df = df.sort_values(["source_file", "_time"]).reset_index(drop=True)

    after = len(df)
    print(f"[data_pipeline] cleaning: {before} -> {after} rows ({before - after} dropped)")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def thermal_risk(proc_temp_c: pd.Series, dTdt: pd.Series,
                  limit_c: float = THERMAL_LIMIT_C, safe_low_c: float = SAFE_LOW_C) -> pd.Series:
    """PLACEHOLDER risk formula -- see module docstring. 0 = safe, 1 = at/over limit."""
    proximity = ((proc_temp_c - safe_low_c) / (limit_c - safe_low_c)).clip(0, 1)
    rate_term = (dTdt.clip(lower=0) / 1.0).clip(0, 1)  # >=1C/sample treated as maximal
    return (0.7 * proximity + 0.3 * rate_term).clip(0, 1)


def pwm_target_heuristic(proc_temp_c: pd.Series, dTdt: pd.Series, cpu_usage: pd.Series,
                          limit_c: float = THERMAL_LIMIT_C, safe_low_c: float = SAFE_LOW_C) -> pd.Series:
    """PLACEHOLDER target for rows with no directly observed PWM (PID/TEAP/
    HyperFan files). Combines thermal safety + temperature trend + workload
    + cooling effort into a 0-100% command, per the spec's own section 6
    description -- but the spec describes the *goal*, not a formula, so
    this specific combination is this pipeline's interpretation, not a
    validated target. Monotonic in all three inputs, capped at 100%."""
    thermal_term = ((proc_temp_c - safe_low_c) / (limit_c - safe_low_c)).clip(0, 1) * 70.0
    trend_term = dTdt.clip(lower=0).clip(upper=1.0) * 20.0
    workload_term = (cpu_usage / 100.0).clip(0, 1) * 10.0
    return (thermal_term + trend_term + workload_term).clip(0, 100)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["proc_temp_c"] = (df["Cpu1_Temp"] + df["Cpu2_Temp"]) / 2.0
    df["inlet_temp_c"] = df["Inlet_Temp"]

    # NOTE: deliberately NOT using groupby(...).apply() here -- on pandas
    # >=2.2 that silently excludes the grouping column ("source_file") from
    # what's passed to the function (the include_groups default), which
    # quietly drops the column this whole pipeline is keyed on downstream.
    # Manual iterate-and-concat keeps every column, including the group key.
    per_file_results = []
    for _, g in df.groupby("source_file"):
        g = g.sort_values("_time").copy()
        dt_s = g["_time"].diff().dt.total_seconds()
        dT = g["proc_temp_c"].diff()
        g["dTdt_c_per_s"] = (dT / dt_s).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        g["future_temp_c"] = g["proc_temp_c"].shift(-FUTURE_HORIZON_ROWS)
        per_file_results.append(g)
    df = pd.concat(per_file_results, ignore_index=True)
    df = df.dropna(subset=["future_temp_c"]).reset_index(drop=True)

    df["thermal_risk_score"] = thermal_risk(df["proc_temp_c"], df["dTdt_c_per_s"])

    is_direct = df["fan_mode_type"] == "FIXED_PWM"
    df["pwm_pct"] = np.where(
        is_direct,
        df["fixed_pwm_pct"],
        pwm_target_heuristic(df["proc_temp_c"], df["dTdt_c_per_s"], df["cpu_usage"]),
    )
    df["pwm_label_source"] = np.where(is_direct, "direct", "heuristic")

    return df


# ---------------------------------------------------------------------------
# Splitting -- grouped by source_file so one experiment never leaks across
# train/val/test (a file is one continuous run, not independent samples).
#
# An earlier version of this function stratified by each file's fan-control
# condition, trying to force every discrete fixed-PWM level into every
# split. That made things WORSE, not better, and it's worth knowing why:
# this dataset's direct PWM supervision is not just imbalanced, it's
# genuinely sparse -- only 4 distinct fixed-PWM values exist across all 122
# files (0%, ~11.8% [PWM 30/255, the overwhelming majority], ~35.3%, 100%).
# Forcing a degenerate single-value validation split (e.g. val = 100%
# samples all at the 11.8% level) makes R2 wildly unstable, since R2's
# denominator is the target's own variance in that split -- near-zero
# variance makes R2 blow up in either direction from small, ordinary
# errors. That's a validation-set artifact, not a worse model.
#
# Plain file-level GroupShuffleSplit (below) doesn't solve the underlying
# sparsity -- nothing can, short of more diverse source data -- but it
# doesn't actively manufacture degenerate splits either, and gives a more
# stable (if still imperfect) read on the PWM head's real performance. See
# README's "Honest results" section for what this means for the PWM output
# specifically.
# ---------------------------------------------------------------------------

def split_by_file(df: pd.DataFrame, test_size=0.2, val_size=0.2, seed=42):
    groups = df["source_file"]

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))
    trainval_df = df.iloc[trainval_idx]
    test_df = df.iloc[test_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(gss2.split(trainval_df, groups=trainval_df["source_file"]))
    train_df = trainval_df.iloc[train_idx]
    val_df = trainval_df.iloc[val_idx]

    print(f"[data_pipeline] split -> train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"(by distinct source files, not row count)")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(df[FEATURE_COLUMNS])


# ---------------------------------------------------------------------------
# Augmentation (train split only, raw feature space, pre-scaling)
# ---------------------------------------------------------------------------

def augment_temperature_perturbation(df: pd.DataFrame, sigma_c=0.3, n_copies=1, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    copies = []
    for _ in range(n_copies):
        c = df.copy()
        c["proc_temp_c"] += rng.normal(0, sigma_c, size=len(c))
        copies.append(c)
    return pd.concat(copies, ignore_index=True)


def augment_inlet_variation(df: pd.DataFrame, delta_range_c=(-1.5, 1.5), n_copies=1, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    copies = []
    for _ in range(n_copies):
        c = df.copy()
        offset = rng.uniform(*delta_range_c)
        c["inlet_temp_c"] += offset
        copies.append(c)
    return pd.concat(copies, ignore_index=True)


def augment_measurement_noise(df: pd.DataFrame, noise_std: dict, n_copies=1, seed=2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    copies = []
    for _ in range(n_copies):
        c = df.copy()
        for col, std in noise_std.items():
            c[col] = c[col] + rng.normal(0, std, size=len(c))
        copies.append(c)
    return pd.concat(copies, ignore_index=True)


def augment_workload_interpolation(df: pd.DataFrame, n_interp=1, seed=3) -> pd.DataFrame:
    """Linearly interpolate between consecutive rows within the same source
    file to fill gaps between the dataset's actual sampling interval."""
    rng = np.random.default_rng(seed)
    synth_rows = []
    numeric_cols = FEATURE_COLUMNS + TARGET_COLUMNS

    for _, g in df.groupby("source_file"):
        g = g.sort_values("_time").reset_index(drop=True)
        for i in range(len(g) - 1):
            for _ in range(n_interp):
                alpha = rng.uniform(0.2, 0.8)
                row = g.iloc[i][numeric_cols] * (1 - alpha) + g.iloc[i + 1][numeric_cols] * alpha
                row["source_file"] = g.iloc[i]["source_file"]
                row["pwm_label_source"] = g.iloc[i]["pwm_label_source"]
                synth_rows.append(row)

    return pd.DataFrame(synth_rows).reset_index(drop=True)


def build_augmented_training_set(train_df: pd.DataFrame) -> pd.DataFrame:
    parts = [
        train_df,
        augment_temperature_perturbation(train_df, n_copies=1),
        augment_inlet_variation(train_df, n_copies=1),
        augment_measurement_noise(train_df, {"dTdt_c_per_s": 0.02, "cpu_usage": 1.0}, n_copies=1),
        augment_workload_interpolation(train_df, n_interp=1),
    ]
    out = pd.concat(parts, ignore_index=True)
    print(f"[data_pipeline] augmentation: {len(train_df)} -> {len(out)} training rows")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../bmcdata_repo/data")
    ap.add_argument("--inventory", default="bmcdata_inventory.csv")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--limit", type=int, default=None,
                     help="load only the first N inventory files (omit for the full ~124-file dataset)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    inv = load_inventory(args.inventory)
    files = inv["file"].head(args.limit).tolist() if args.limit else None
    df = load_dataset(args.data_dir, args.inventory, files=files)

    df = clean_data(df)
    df = engineer_features(df)
    print(f"[data_pipeline] pwm_label_source counts:\n{df['pwm_label_source'].value_counts()}")

    train_df, val_df, test_df = split_by_file(df)
    train_aug = build_augmented_training_set(train_df)

    keep_cols = FEATURE_COLUMNS + TARGET_COLUMNS + ["source_file", "pwm_label_source"]
    train_aug[keep_cols].to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    val_df[keep_cols].to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    test_df[keep_cols].to_csv(os.path.join(args.out_dir, "test.csv"), index=False)
    print(f"[data_pipeline] wrote train/val/test CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()

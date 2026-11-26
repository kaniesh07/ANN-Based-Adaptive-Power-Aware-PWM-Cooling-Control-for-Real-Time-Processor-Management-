"""
visualize.py

Plotting functions for the charts the spec's evaluation section (11) asks
for. Takes DataFrames (processed CSVs, or predictions merged with ground
truth) -- nothing runs standalone until you have real results to hand it.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] wrote {out_path}")


def plot_temp_vs_time(df: pd.DataFrame, out_path: str, source_file: str = None):
    d = df[df["source_file"] == source_file] if source_file else df
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pd.to_datetime(d["_time"]) if "_time" in d else range(len(d)), d["proc_temp_c"], label="proc_temp_c")
    ax.plot(pd.to_datetime(d["_time"]) if "_time" in d else range(len(d)), d["inlet_temp_c"], label="inlet_temp_c")
    ax.set_xlabel("time"); ax.set_ylabel("deg C"); ax.legend()
    ax.set_title("Temperature vs. Time")
    _save(fig, out_path)


def plot_pwm_vs_time(df: pd.DataFrame, out_path: str, source_file: str = None):
    d = df[df["source_file"] == source_file] if source_file else df
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(d)), d["pwm_pct"])
    ax.set_xlabel("sample"); ax.set_ylabel("PWM (%)")
    ax.set_title("PWM vs. Time")
    _save(fig, out_path)


def plot_workload_vs_temp(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["cpu_usage"], df["proc_temp_c"], s=4, alpha=0.3)
    ax.set_xlabel("cpu_usage (%)"); ax.set_ylabel("proc_temp_c")
    ax.set_title("Workload vs. Temperature")
    _save(fig, out_path)


def plot_dTdt_vs_pwm(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["pwm_pct"], df["dTdt_c_per_s"], s=4, alpha=0.3)
    ax.set_xlabel("PWM (%)"); ax.set_ylabel("dT/dt (C/s)")
    ax.set_title("Temperature Rate-of-Rise vs. PWM")
    _save(fig, out_path)


def plot_predicted_vs_actual(y_true, y_pred, label: str, out_path: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.3)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel(f"actual {label}"); ax.set_ylabel(f"predicted {label}")
    ax.set_title(f"Predicted vs. Actual {label}")
    _save(fig, out_path)


def plot_fan_mode_breakdown(inventory_df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    inventory_df["fan_mode_type"].value_counts().plot(kind="bar", ax=ax)
    ax.set_ylabel("file count")
    ax.set_title("BMCDATA files by fan_mode_type\n(only FIXED_PWM gives a direct PWM label)")
    _save(fig, out_path)


def plot_float_vs_int8_accuracy(metrics_float: dict, metrics_int8: dict, target: str, out_path: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    keys = ["MAE", "RMSE"]
    x = range(len(keys))
    ax.bar([i - 0.15 for i in x], [metrics_float[target][k] for k in keys], width=0.3, label="float")
    ax.bar([i + 0.15 for i in x], [metrics_int8[target][k] for k in keys], width=0.3, label="INT8")
    ax.set_xticks(list(x)); ax.set_xticklabels(keys)
    ax.set_title(f"Float vs INT8 accuracy: {target}"); ax.legend()
    _save(fig, out_path)


def plot_control_comparison(df_ann: pd.DataFrame, df_fixed: pd.DataFrame,
                             df_rulebased: pd.DataFrame, out_path: str):
    """Overlay proc_temp_c across the three firmware control modes, run
    under matched workload conditions (see config.h's ACTIVE_CONTROL_MODE)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, d in [("ANN", df_ann), ("Fixed-PWM", df_fixed), ("Rule-based", df_rulebased)]:
        ax.plot(range(len(d)), d["proc_temp_c"], label=label)
    ax.set_xlabel("sample"); ax.set_ylabel("proc_temp_c"); ax.legend()
    ax.set_title("Controller Comparison: Temperature Response")
    _save(fig, out_path)


def plot_thermal_limit_violations(df_by_controller: dict, limit_c: float, out_path: str):
    labels, counts = [], []
    for name, d in df_by_controller.items():
        labels.append(name)
        counts.append(int((d["proc_temp_c"] >= limit_c).sum()))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, counts)
    ax.set_ylabel(f"# samples >= {limit_c}C")
    ax.set_title("Thermal-Limit Violations by Controller")
    _save(fig, out_path)

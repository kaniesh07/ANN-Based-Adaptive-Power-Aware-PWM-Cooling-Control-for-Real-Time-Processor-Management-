"""
train.py

Trains the baseline floating-point ANN on data_pipeline.py's processed
train/val/test CSVs. Rows whose PWM label came from the heuristic fallback
(pwm_label_source == "heuristic") are downweighted relative to directly
observed FIXED_PWM rows, so the network isn't trained to trust a synthetic
target as much as a real one. future_temp_c/thermal_risk_score use full
weight for every row regardless of label source, since those don't depend
on the PWM-observability issue.

Usage:
    python train.py --data-dir data/processed --out-dir models
"""

import argparse
import json
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # must precede any tensorflow import -- see model.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_pipeline import FEATURE_COLUMNS, TARGET_COLUMNS, fit_scaler, apply_scaler
from model import build_model, compile_model

import tensorflow as tf

HEURISTIC_PWM_WEIGHT = 0.3  # how much to trust heuristic-labeled PWM rows vs direct (1.0)


def load_split(data_dir: str, name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, f"{name}.csv"))


def build_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """A single scalar weight per row (Keras applies one weight across all
    three output dims per sample).

    Two adjustments combined:
      1. heuristic-labeled rows get downweighted overall (HEURISTIC_PWM_WEIGHT)
      2. WITHIN direct-labeled rows, inverse-frequency weighting by PWM
         bucket -- this dataset's direct PWM supervision is overwhelmingly
         one value (PWM 30/255, ~11.8%), so without this the loss is
         dominated by that single level and the model has little incentive
         to get the rarer 0%/35%/100% examples right. This does NOT fix the
         underlying data sparsity (only 4 distinct direct PWM values exist
         in the whole dataset) -- it just stops training from ignoring the
         minority ones entirely. See README for what this does and doesn't
         fix."""
    base = np.where(df["pwm_label_source"].values == "direct", 1.0, HEURISTIC_PWM_WEIGHT)

    is_direct = df["pwm_label_source"].values == "direct"
    if is_direct.any():
        pwm_bucket = df["pwm_pct"].round(0).values
        direct_buckets = pwm_bucket[is_direct]
        counts = pd.Series(direct_buckets).value_counts()
        # Clipped to 5x: the raw inverse-frequency ratio between the
        # dominant PWM-30/255 bucket (~68k rows) and the rarest bucket
        # (~360 rows) is close to 200x -- applying that directly overwhelms
        # the shared per-sample weight (it multiplies ALL three output
        # heads' loss, not just PWM's), which measurably wrecked the
        # future_temp/thermal_risk heads' accuracy in testing without
        # meaningfully fixing the PWM head either. A capped multiplier is a
        # partial, honest mitigation, not a fix for the underlying sparsity.
        inv_freq = (counts.max() / counts).clip(upper=5.0)
        bucket_weight = pd.Series(pwm_bucket).map(inv_freq).fillna(1.0).values
        base = np.where(is_direct, base * bucket_weight, base)

    return base


def evaluate(model, X, y, split_name: str, mask=None) -> dict:
    y_pred = model.predict(X, verbose=0)
    # model output[:,0] is a 0-1 pwm_frac (see model.py docstring) -- rescale
    # to 0-100 so reported PWM MAE/RMSE are in percentage points, comparable
    # to the pwm_pct values in y_true.
    y_pred = y_pred.copy()
    y_pred[:, 0] *= 100.0
    if mask is not None:
        y = y[mask]
        y_pred = y_pred[mask]
    metrics = {}
    for i, col in enumerate(TARGET_COLUMNS):
        yt, yp = y[:, i], y_pred[:, i]
        metrics[col] = {
            "MAE": float(mean_absolute_error(yt, yp)),
            "MSE": float(mean_squared_error(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "R2": float(r2_score(yt, yp)) if len(yt) > 1 else float("nan"),
        }
    print(f"[train] {split_name} metrics ({len(y)} rows):")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_df = load_split(args.data_dir, "train")
    val_df = load_split(args.data_dir, "val")
    test_df = load_split(args.data_dir, "test")

    scaler = fit_scaler(train_df)
    X_train = apply_scaler(train_df, scaler)
    X_val = apply_scaler(val_df, scaler)
    X_test = apply_scaler(test_df, scaler)

    y_train = train_df[TARGET_COLUMNS].values
    y_val = val_df[TARGET_COLUMNS].values
    y_test = test_df[TARGET_COLUMNS].values

    sw_train = build_sample_weights(train_df)

    model = build_model()
    model = compile_model(model)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.out_dir, "baseline_best.keras"), monitor="val_loss", save_best_only=True
        ),
    ]

    model.fit(
        X_train, y_train,
        sample_weight=sw_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    evaluate(model, X_val, y_val, "validation (all rows)")
    evaluate(model, X_test, y_test, "test (all rows)")

    direct_mask = (test_df["pwm_label_source"] == "direct").values
    if direct_mask.any():
        evaluate(model, X_test, y_test, "test (direct-PWM rows only -- the defensible number)", mask=direct_mask)

    model.save(os.path.join(args.out_dir, "baseline_float.keras"))

    scaler_params = {
        "features": FEATURE_COLUMNS,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(os.path.join(args.out_dir, "scaler_params.json"), "w") as f:
        json.dump(scaler_params, f, indent=2)

    print(f"[train] saved baseline model + scaler params to {args.out_dir}")


if __name__ == "__main__":
    main()

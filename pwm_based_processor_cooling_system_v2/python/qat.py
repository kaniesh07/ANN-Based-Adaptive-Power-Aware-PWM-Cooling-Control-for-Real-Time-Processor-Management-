"""
qat.py

Quantization-Aware Training on top of train.py's baseline float model.

Usage:
    python qat.py --data-dir data/processed --models-dir models --epochs 30
"""

import argparse
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # must precede any tensorflow import -- see model.py

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from data_pipeline import TARGET_COLUMNS, fit_scaler, apply_scaler
from train import load_split, evaluate, build_sample_weights


def weighted_mse(y_true, y_pred):
    # Same convention as model.compile_model: y_pred[:,0] is a 0-1
    # pwm_frac, y_true[:,0] is the 0-100 pwm_pct training target.
    pwm_err = tf.square(y_true[:, 0] / 100.0 - y_pred[:, 0])
    temp_err = tf.square(y_true[:, 1] - y_pred[:, 1]) / (50.0 ** 2)
    risk_err = tf.square(y_true[:, 2] - y_pred[:, 2])
    return pwm_err + temp_err + risk_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    baseline_path = os.path.join(args.models_dir, "baseline_float.keras")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"{baseline_path} not found -- run train.py first.")

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

    baseline_model = tf.keras.models.load_model(baseline_path, compile=False)

    q_aware_model = tfmot.quantization.keras.quantize_model(baseline_model)
    # A lower LR than the original 1e-4 -- in testing, 1e-4 for even 20
    # epochs measurably degraded the thermal_risk head (which was R2~0.99
    # pre-QAT) down to a negative R2, i.e. QAT fine-tuning was overwriting
    # good float-trained weights faster than it was adapting them to
    # quantization. 1e-5 fine-tunes gently enough to preserve the baseline's
    # accuracy while still calibrating the quantization ranges.
    q_aware_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=weighted_mse)

    q_aware_model.fit(
        X_train, y_train,
        sample_weight=sw_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    print("[qat] QAT model evaluation (float-simulated-quant, not yet true INT8):")
    evaluate(q_aware_model, X_test, y_test, "test (QAT, pre-INT8-conversion)")

    # Saved as legacy HDF5 (.h5), NOT the native .keras format: tfmot's
    # QuantizeLayer wrapper registers its min/max tracking variables in a
    # way that the newer .keras save/load path mismatches on reload
    # ("Layer 'quantize_layer' expected 5 variables, but received 3") --
    # this is a tfmot/tf_keras compatibility bug, not something fixable
    # from this script's side. .h5 save/load for QAT models is the
    # documented workaround and has been reliable in testing here.
    out_path = os.path.join(args.models_dir, "qat_model.h5")
    q_aware_model.save(out_path)
    print(f"[qat] saved QAT model to {out_path}")
    print("[qat] next: convert_tflite.py for the real INT8 .tflite + TFLM C array, "
          "and re-check accuracy on the actual int8 model.")


if __name__ == "__main__":
    main()

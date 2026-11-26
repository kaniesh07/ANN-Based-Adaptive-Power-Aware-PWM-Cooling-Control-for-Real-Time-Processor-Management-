"""
convert_tflite.py

Converts the QAT Keras model into a full-INT8 .tflite file, evaluates the
REAL int8 model's accuracy vs the float baseline, and emits a C byte array
(model_data.h/.cc) for TensorFlow Lite Micro on the ESP32.

Usage:
    python convert_tflite.py --data-dir data/processed --models-dir models
"""

import argparse
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # must precede any tensorflow import -- see model.py

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from data_pipeline import TARGET_COLUMNS, fit_scaler, apply_scaler
from train import load_split


def representative_dataset_gen(X_train: np.ndarray):
    def gen():
        n = min(300, len(X_train))
        idx = np.random.default_rng(0).choice(len(X_train), size=n, replace=False)
        for i in idx:
            yield [X_train[i:i + 1].astype(np.float32)]
    return gen


def convert_to_int8_tflite(keras_model_path: str, X_train: np.ndarray) -> bytes:
    # A QAT-saved model contains tfmot's custom Quantize* wrapper layers --
    # loading it back without quantize_scope() fails deep inside Keras's
    # functional-model reconstruction (a confusing "'str' object is not
    # callable" with no mention of quantization anywhere in the traceback).
    with tfmot.quantization.keras.quantize_scope():
        model = tf.keras.models.load_model(keras_model_path, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()


def evaluate_tflite_int8(tflite_bytes: bytes, X_test: np.ndarray, y_test: np.ndarray):
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale, in_zero = input_details["quantization"]
    out_scale, out_zero = output_details["quantization"]

    preds = []
    for row in X_test:
        q_in = np.round(row / in_scale + in_zero).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], q_in.reshape(1, -1))
        interpreter.invoke()
        q_out = interpreter.get_tensor(output_details["index"])[0]
        preds.append((q_out.astype(np.float32) - out_zero) * out_scale)

    preds = np.array(preds)
    preds[:, 0] *= 100.0  # pwm_frac (0-1) -> pwm_pct, same convention as train.py's evaluate()

    from sklearn.metrics import mean_absolute_error
    for i, col in enumerate(TARGET_COLUMNS):
        mae = mean_absolute_error(y_test[:, i], preds[:, i])
        print(f"[convert_tflite] INT8 test MAE for {col}: {mae:.4f}")
    return preds


def write_c_array(tflite_bytes: bytes, out_dir: str, var_name: str = "g_cooling_ann_model"):
    header_path = os.path.join(out_dir, "model_data.h")
    source_path = os.path.join(out_dir, "model_data.cc")

    with open(header_path, "w") as f:
        f.write("#pragma once\n#include <cstdint>\n\n")
        f.write(f"extern const unsigned char {var_name}[];\n")
        f.write(f"extern const unsigned int {var_name}_len;\n")

    with open(source_path, "w") as f:
        f.write(f'#include "model_data.h"\n\n')
        f.write(f"const unsigned char {var_name}[] = {{\n")
        for i, b in enumerate(tflite_bytes):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{b:02x}, ")
            if i % 12 == 11:
                f.write("\n")
        f.write("\n};\n")
        f.write(f"const unsigned int {var_name}_len = {len(tflite_bytes)};\n")

    print(f"[convert_tflite] wrote {header_path} and {source_path} "
          f"({len(tflite_bytes)} bytes) -- copy both into firmware/src/.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    qat_path = os.path.join(args.models_dir, "qat_model.h5")  # see qat.py's save-format note
    if not os.path.exists(qat_path):
        raise FileNotFoundError(f"{qat_path} not found -- run qat.py first.")

    train_df = load_split(args.data_dir, "train")
    test_df = load_split(args.data_dir, "test")
    scaler = fit_scaler(train_df)
    X_train = apply_scaler(train_df, scaler)
    X_test = apply_scaler(test_df, scaler)
    y_test = test_df[TARGET_COLUMNS].values

    tflite_bytes = convert_to_int8_tflite(qat_path, X_train)

    tflite_path = os.path.join(args.models_dir, "cooling_ann_int8.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"[convert_tflite] wrote {tflite_path} ({len(tflite_bytes)} bytes)")

    evaluate_tflite_int8(tflite_bytes, X_test, y_test)
    write_c_array(tflite_bytes, args.models_dir)


if __name__ == "__main__":
    main()

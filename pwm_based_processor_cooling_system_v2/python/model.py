"""
model.py

4 -> 32 -> 32 -> 16 -> 16 -> 8 -> 3, per the spec (five hidden Dense layers
followed by a three-neuron output layer -- unambiguous this time, unlike
the physical-acquisition version of this project, which had a genuine
4-vs-5-input contradiction in its source doc).

Inputs (data_pipeline.FEATURE_COLUMNS):
    0: proc_temp_c    -- avg(Cpu1_Temp, Cpu2_Temp)
    1: dTdt_c_per_s
    2: inlet_temp_c
    3: cpu_usage       -- real measured workload, 0-100

Outputs (single Dense(3)):
    0: pwm_frac       (0-1, sigmoid -- NOT pre-scaled to 0-100 inside the
                        graph; see note below)
    1: future_temp_c  (linear)
    2: thermal_risk    (0-1, sigmoid)

On output[0] staying as a 0-1 fraction rather than a 0-100 percentage: an
earlier version of this model multiplied the sigmoid output by 100 inside
the graph (a Lambda layer, then a Rescaling layer after Lambda turned out
to be unsafe to deserialize). Both failed for QAT: a raw-lambda Lambda
layer won't deserialize from a saved .keras file without disabling a
safety check, and tfmot's quantize_model() doesn't support Rescaling at
all ("Layer pwm_pct:...Rescaling is not supported"). Keeping the output as
a plain sigmoid (same shape as thermal_risk) sidesteps both failure modes
and is the convention this whole pipeline follows: wherever you read this
model's first output -- train.py's evaluate(), convert_tflite.py's INT8
evaluation, and the ESP32 firmware's tflite_inference.cpp -- multiply it
by 100 yourself to get a PWM percentage. Don't reintroduce an in-graph
scaling layer without re-testing it survives qat.py end-to-end.

NOTE: forces the legacy tf_keras (Keras 2) API via TF_USE_LEGACY_KERAS,
set BEFORE importing tensorflow anywhere in this pipeline. As of writing,
tensorflow-model-optimization (qat.py's quantize_model) does not support
Keras 3's model classes -- tfmot.quantize_model() raises "`to_quantize`
can only either be a keras Sequential or Functional model" against a
perfectly good Functional model, because it's checking for the old
tf_keras.Model class. Every entry-point script in this pipeline
(model.py, train.py, qat.py, convert_tflite.py) sets this at the very top
for consistency -- all of them need to agree on which Keras a saved
.keras file was written with, or loading breaks across scripts.
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow.keras import layers, models

INPUT_DIM = 4
HIDDEN_UNITS = [32, 32, 16, 16, 8]


def build_model(input_dim: int = INPUT_DIM) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_dim,), name="features")

    x = inputs
    for i, units in enumerate(HIDDEN_UNITS):
        x = layers.Dense(units, activation="relu", name=f"dense_{i+1}")(x)

    # Stays a 0-1 fraction, not pre-scaled to 0-100 -- see module docstring
    # for why (QAT compatibility).
    pwm_frac = layers.Dense(1, activation="sigmoid", name="pwm_head")(x)

    future_temp_c = layers.Dense(1, activation="linear", name="future_temp_c")(x)
    thermal_risk = layers.Dense(1, activation="sigmoid", name="thermal_risk")(x)

    outputs = layers.Concatenate(name="outputs")([pwm_frac, future_temp_c, thermal_risk])

    model = models.Model(inputs=inputs, outputs=outputs, name="cooling_ann")
    return model


def compile_model(model: tf.keras.Model, lr: float = 1e-3) -> tf.keras.Model:
    """Per-row sample weights (see train.py) downweight heuristic-labeled
    PWM targets -- so the loss itself stays a plain weighted-MSE across the
    three outputs; the direct/heuristic distinction is handled via Keras's
    sample_weight mechanism at fit() time, not inside this loss fn."""
    def weighted_mse(y_true, y_pred):
        # y_true[:, 0] arrives as a 0-100 pwm_pct target (that's what's in
        # the training CSV); y_pred[:, 0] is the model's 0-1 pwm_frac
        # output -- divide the target by 100 here so both sides compare in
        # the same 0-1 units.
        pwm_err = tf.square(y_true[:, 0] / 100.0 - y_pred[:, 0])
        temp_err = tf.square(y_true[:, 1] - y_pred[:, 1]) / (50.0 ** 2)
        risk_err = tf.square(y_true[:, 2] - y_pred[:, 2])
        return pwm_err + temp_err + risk_err  # per-row, NOT reduced -- sample_weight needs per-row loss

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=weighted_mse)
    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
    print(f"Total trainable parameters: {m.count_params()}")

# ANN-Based Adaptive Power-Aware PWM Cooling Control (BMCDATA version)

This is the reconstructed project: no physical data-acquisition phase. The
ANN is trained offline on the real, public **BMCDATA** dataset
(https://github.com/arealuser/bmcdata, DOI 10.21227/6yep-by05 on IEEE
DataPort). The ESP32 firmware here does inference + control only.

Everything in `python/` was actually run against the real cloned dataset
while building this (not just written and hoped-to-work) — see "What was
verified" below for exactly what that covered, and the things that broke
and got fixed along the way, since those are worth knowing before you build
on top of this.

## Directory layout

```
python/
  bmcdata_inventory.csv   per-file experiment metadata (fan mode, workload),
                          copied from the dataset's own README table -- this
                          is NOT inside the CSV data files themselves
  bmcdata_loader.py       parses the real InfluxDB-exported CSV format
  data_pipeline.py        cleaning, feature engineering, PWM-label
                          construction, split, train-only augmentation
  model.py                4->32->32->16->16->8->3 Keras architecture
  train.py                baseline float training, sample-weighted
  qat.py                  quantization-aware fine-tuning
  convert_tflite.py       INT8 .tflite + C array export for TFLite Micro
  visualize.py            plotting functions for the spec's evaluation charts
  requirements.txt

firmware/                 PlatformIO project (ESP32-WROOM-32)
  src/
    config.h               pins, ACTIVE_CONTROL_MODE switch, thermal limit
    sensors.h/.cpp          DS18B20 x2 + INA219 (same hardware as before)
    fan_control.h/.cpp       LEDC PWM -> IRLZ44N gate
    workload.h/.cpp          workload PROXY (see honesty note below)
    tflite_inference.h/.cpp  loads + runs the trained INT8 model, live
    rule_based_control.h/.cpp  baseline controller for comparison runs
    logger.h/.cpp             CSV-over-serial, now logs all 3 control
                               modes' relevant fields in one schema
    main.cpp                  real-time control loop
    model_data.h/.cc          placeholder -- convert_tflite.py overwrites
```

## Running the offline pipeline

```
git clone https://github.com/arealuser/bmcdata.git
cd pwm_cooling_project_v2/python
pip install -r requirements.txt

python data_pipeline.py --data-dir ../../bmcdata/data --inventory bmcdata_inventory.csv --out-dir data/processed
python train.py --data-dir data/processed --out-dir models
python qat.py --data-dir data/processed --models-dir models
python convert_tflite.py --data-dir data/processed --models-dir models
```

Then copy `models/model_data.h` and `models/model_data.cc` into
`firmware/src/`, copy the `mean`/`scale` arrays from
`models/scaler_params.json` into `tflite_inference.cpp`'s
`SCALER_MEAN`/`SCALER_SCALE`, set `ACTIVE_CONTROL_MODE` in `config.h`, and
flash.

By default `data_pipeline.py` loads every file in the inventory (~35MB,
124 files, a few minutes). Pass `--limit 15` while iterating to work with
a fast subset first.

## What was verified (not just written)

Every Python script above was actually run end-to-end against the real
cloned dataset (not synthetic/sample data) while building this, and each
of these was a real bug caught and fixed in that process — worth knowing
about since they'll bite you again if you restructure the code:

1. **The dataset's own column naming is a trap.** `_measurement` is a
   constant project codename (`"sdgp"`) in every file — it's
   `device_id` (`"bmc"` vs `"host"`) that actually distinguishes the two
   data sections. Keying a loader off `_measurement`, which is the more
   "obvious" column name, silently merges both sections together.
2. **One real file (`202306172153.csv`) has a malformed row** that
   coerces to a null `_time` — `merge_asof` hard-fails on a null join key,
   so the loader drops null-`_time` rows before merging rather than
   failing that whole file.
3. **PWM is NOT a per-row column anywhere in this dataset.** `FAN1`-`FAN4`
   are fan *speeds* (RPM), not duty cycle. The actual commanded PWM is
   file-level metadata (`bmcdata_inventory.csv`, parsed from the dataset's
   README table) and is only a clean, directly-observed label for files
   run under a single fixed PWM value. PID/TEAP/HyperFan files have no
   directly observable PWM in this dataset at all — `data_pipeline.py`
   falls back to a heuristic target for those rows (see its docstring) and
   `train.py` downweights them during training. **If you report a PWM
   accuracy number, report it on direct-label rows only** — `train.py`'s
   evaluate() already does this as a separate printed block.
4. **A `Lambda` layer (`x * 100`) fails to deserialize from a saved
   `.keras` file** by default (Keras treats arbitrary Python lambdas in a
   saved file as a code-execution risk). Swapping it for `Rescaling` fixed
   loading, but then **`tfmot.quantize_model()` doesn't support
   `Rescaling` either** ("Layer pwm_pct:...Rescaling is not supported").
   The actual fix: the model's PWM head just outputs a plain 0-1 sigmoid
   fraction, same shape as the risk head, and every consumer (eval code,
   TFLite postprocessing, firmware) multiplies by 100 itself. See
   `model.py`'s docstring.
5. **`tensorflow-model-optimization` 0.8.1 does not support Keras 3**
   model classes — `quantize_model()` on a perfectly valid Functional
   model raises `"to_quantize can only either be a keras Sequential or
   Functional model"`. Fix: force the legacy `tf_keras` (Keras 2) API via
   `TF_USE_LEGACY_KERAS=1`, set before importing `tensorflow` anywhere —
   every entry-point script does this at the top. All scripts need to
   agree on this, or loading a file saved by one script from another
   breaks.
6. **Saving a QAT model in the native `.keras` format and reloading it
   fails** (`"Layer 'quantize_layer' expected 5 variables, but received
   3"`) — a tfmot/tf_keras save-format compatibility bug, not something
   fixable from calling code. `qat.py` saves as legacy `.h5` instead,
   which round-trips correctly; `convert_tflite.py` loads the `.h5` inside
   `tfmot.quantization.keras.quantize_scope()` (needed separately, or
   loading fails with an unrelated-looking `'str' object is not
   callable`).

With those fixes in place, the full pipeline (load → clean → engineer →
split → augment → train → QAT → INT8 convert → C array) ran cleanly on a
15-file subset of the real dataset, producing a working 9KB INT8 model.
Metrics on a 1-2 epoch smoke-test run are meaningless (not enough training)
— that's expected, and not something to read into; run the full ~124-file
dataset for a real number.

## Hardware (unchanged from the physical-acquisition version)

Same BOM, same wiring (DS18B20 x2 on GPIO4 + 4.7kΩ pull-up, INA219 on I2C,
IRLZ44N fan drive on GPIO25, filter caps on the MOSFET drain). This
firmware uses the sensors for **real-time control and validation**, not
for generating training data — the ANN never sees this hardware's own
measurements during training, only at inference time.

## Running without the INA219

`USE_INA219` in `firmware/src/config.h` is set to `0` in this build. With it
off, the firmware never touches I2C (no `Error -1` spam), and the
`fan_voltage_v`, `fan_current_ma` and `fan_power_mw` log columns read 0.

Wiring without it: fan (+) goes straight to the fan supply rail; fan (-) goes
to the MOSFET Drain; everything else is unchanged. GPIO21/22 are unused.

Cooling power and energy (spec section 11) then need manual measurement:
log the fan's supply voltage and current with the multimeter or DC power meter
at several fixed PWM levels, build a power-vs-PWM table, and estimate each
controller run's power and energy from its logged PWM values. That is an
estimate, not per-sample measurement; say so in the write-up. Set
`USE_INA219 1` if you wire the module back in.

## Honest results (full 122-file dataset, actually trained)

This was run end-to-end on the real, complete BMCDATA dataset (122 of the
124 inventoried files were present in the public repo clone; 2 missing).
`firmware/src/model_data.h/.cc` already contain this trained model, and
`tflite_inference.cpp`'s scaler constants are already set from the real
`scaler_params.json` — nothing further to train before flashing, though
see the caveat below before trusting the PWM output specifically.

**Real INT8 model, test-set MAE:**

| Output | Test MAE | Trustworthy? |
|---|---|---|
| `future_temp_c` | 0.57 °C | Yes — strong, consistent across every configuration tried |
| `thermal_risk_score` (0-1 scale) | 0.05 | Reasonable in absolute terms |
| `pwm_pct` | 12.4 percentage points | **No — see below** |

**The PWM head is not reliable, and this is a dataset limitation, not a
bug left unfixed.** Across the whole 122-file dataset, only **4 distinct
PWM values** are ever directly observed (the dataset is overwhelmingly
dominated by files run at PWM 30/255 ≈ 11.8%; 0%, ~35.3%, and 100% each
appear in only a handful of files). Three different mitigations were
tried and measured, not just considered:
- Stratifying the train/val/test split by PWM level — made it *worse*
  (produced a near-single-value validation split, which makes R2
  pathologically unstable)
- Downweighting heuristic-labeled rows relative to direct ones — the
  baseline approach, kept
- Inverse-frequency sample weighting to upweight rare PWM levels — capped
  at 5x, included in the final model; helped the other two heads recover
  but did not fix PWM's test-set generalization

None of these addressed the actual cause: there simply isn't enough
*distinct* PWM supervision in this dataset for a regression head to learn
a reliable general mapping. More files at a wider range of fixed PWM
values would fix this; no amount of pipeline engineering on the existing
122 files does.

**Practical recommendation, reflected in the firmware's default:** don't
let the ANN's PWM output drive the fan unattended yet.
`rule_based_control.cpp`'s simple threshold controller is the safer
choice for `ACTIVE_CONTROL_MODE` today. The firmware runs ANN inference
every tick regardless of which controller is actually driving the fan, so
you still get `future_temp_c`/`thermal_risk_score` — the two outputs that
*are* reliable — logged for monitoring even while a baseline controller
handles actuation.

**QAT note:** quantization-aware fine-tuning at the originally-planned
learning rate (1e-4) measurably degraded the already-good
`thermal_risk_score` head (R2 dropped from ~0.94 to negative) — the model
is small enough that QAT's fake-quant gradients can overwrite good
float-trained weights faster than they adapt them. Dropping to 1e-5
recovered `future_temp_c` but not fully `thermal_risk_score` in testing.
If you retrain, watch all three heads' metrics after QAT, not just after
the float baseline — a QAT pass that quietly breaks one head is an easy
thing to ship by accident.

## The honesty section

1. **Workload is a proxy, not a measurement.** BMCDATA's `cpu_usage`
   feature is real OS-reported CPU utilization on a 128-core server. This
   ESP32 has no equivalent to read — `workload.cpp`'s busy-loop duty
   fraction (`WORKLOAD_BUSY_FRACTION`, 20/55/90%) stands in for it. The
   model was trained on real utilization values; feeding it a proxy at
   inference time means this hardware validates the **control logic**
   the ANN learned, not a faithful re-creation of server-scale thermal
   dynamics on ESP32-scale hardware. Say this plainly in any write-up
   rather than letting a reader assume the ESP32 is "running the same
   workload" the training data describes.
2. **`THERMAL_LIMIT_C` (85°C) must match between `config.h` and
   `data_pipeline.py`.** They're independent constants right now — if you
   tune one without the other, the firmware's thermal_risk output stops
   meaning what it meant during training.
3. **The PWM heuristic-label fallback (data_pipeline.py's
   `pwm_target_heuristic()`) is this pipeline's interpretation of the
   spec's section 6, not a validated optimal-control formula** — the spec
   describes a goal ("thermal safety + temperature trend + workload +
   cooling effort"), not a formula. Treat it as a stand-in for the ~19%
   of rows (PID/TEAP/HyperFan files) with no directly observed PWM, and
   lean on the direct-label evaluation split in `train.py` for any number
   you'd want to defend.
4. **`rule_based_control.cpp`'s threshold curve is deliberately simple**
   (temperature-only, linear ramp) so the ANN-vs-rule-based comparison in
   your evaluation demonstrates something real — what workload- and
   trend-awareness actually buys you — rather than comparing against a
   strawman or a second disguised ANN.

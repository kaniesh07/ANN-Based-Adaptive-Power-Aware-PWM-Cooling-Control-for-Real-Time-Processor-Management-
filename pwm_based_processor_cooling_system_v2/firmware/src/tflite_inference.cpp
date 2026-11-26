// TFLite Micro inference implementation -- 4-input BMCDATA-trained model.
//
// Requires a TFLite Micro Arduino library (e.g. "tanakamasayuki/TensorFlowLite_ESP32"
// or the official TFLM Arduino port). NOT in platformio.ini's lib_deps by
// default -- add it once model_data.h/model_data.cc hold a real trained
// model (python/convert_tflite.py overwrites the placeholders checked into
// this repo):
//
//   pio pkg install --library "tanakamasayuki/TensorFlowLite_ESP32"

#include "tflite_inference.h"
#include "model_data.h"

#if __has_include(<tensorflow/lite/micro/micro_interpreter.h>)
  #define TFLM_AVAILABLE 1
  #include <tensorflow/lite/micro/micro_interpreter.h>
  #include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
  #include <tensorflow/lite/micro/micro_log.h>
  #include <tensorflow/lite/schema/schema_generated.h>
#else
  #define TFLM_AVAILABLE 0
#endif

// -----------------------------------------------------------------------
// Scaler parameters: COPY THESE FROM models/scaler_params.json after
// running python/train.py. Order matches data_pipeline.FEATURE_COLUMNS:
// [proc_temp_c, dTdt_c_per_s, inlet_temp_c, cpu_usage]. Placeholder
// identity values (mean=0, scale=1) below so the build compiles cleanly
// before that file exists.
// -----------------------------------------------------------------------
// Real values from models/scaler_params.json, trained on the full 122-file
// BMCDATA dataset. Order: [proc_temp_c, dTdt_c_per_s, inlet_temp_c, cpu_usage]
static const float SCALER_MEAN[4]  = {45.388822f, 0.000892f, 38.797244f, 17.515153f};
static const float SCALER_SCALE[4] = {4.948011f, 0.021866f, 3.786225f, 21.425386f};

#if TFLM_AVAILABLE

constexpr int kTensorArenaSize = 16 * 1024; // start here, grow if AllocateTensors fails
static uint8_t g_tensor_arena[kTensorArenaSize];

static const tflite::Model* g_model = nullptr;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor* g_input = nullptr;
static TfLiteTensor* g_output = nullptr;

static tflite::MicroMutableOpResolver<4> g_resolver;

bool tflite_inference_begin() {
    g_model = tflite::GetModel(g_cooling_ann_model);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model schema version mismatch: %d vs supported %d",
                     g_model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    g_resolver.AddFullyConnected();
    g_resolver.AddLogistic();   // sigmoid heads (pwm_head, thermal_risk)
    g_resolver.AddConcatenation();
    g_resolver.AddQuantize();

    static tflite::MicroInterpreter static_interpreter(
        g_model, g_resolver, g_tensor_arena, kTensorArenaSize);
    g_interpreter = &static_interpreter;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed -- try increasing kTensorArenaSize");
        return false;
    }

    g_input = g_interpreter->input(0);
    g_output = g_interpreter->output(0);
    return true;
}

AnnOutput tflite_infer(const AnnInput& in) {
    AnnOutput out{};
    if (g_interpreter == nullptr) {
        out.valid = false;
        return out;
    }

    float raw[4] = {in.proc_temp_c, in.dTdt_c_per_s, in.inlet_temp_c, in.cpu_usage};
    float scaled[4];
    for (int i = 0; i < 4; i++) {
        scaled[i] = (raw[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
    }

    float in_scale = g_input->params.scale;
    int32_t in_zero = g_input->params.zero_point;
    for (int i = 0; i < 4; i++) {
        // FIX: round and clamp. A bare (int8_t) cast truncates, and wraps
        // around for out-of-range inputs (e.g. -127 C from a missing sensor
        // becomes a large POSITIVE value).
        long q = lroundf(scaled[i] / in_scale) + in_zero;
        if (q < -128) q = -128;
        if (q > 127)  q = 127;
        g_input->data.int8[i] = (int8_t)q;
    }

    if (g_interpreter->Invoke() != kTfLiteOk) {
        out.valid = false;
        return out;
    }

    float out_scale = g_output->params.scale;
    int32_t out_zero = g_output->params.zero_point;
    float pwm_frac    = (g_output->data.int8[0] - out_zero) * out_scale;
    out.future_temp_c = (g_output->data.int8[1] - out_zero) * out_scale;
    out.thermal_risk  = (g_output->data.int8[2] - out_zero) * out_scale;
    out.pwm_pct = pwm_frac * 100.0f;  // model outputs a 0-1 fraction -- see model.py
    out.valid = true;
    return out;
}

#else // !TFLM_AVAILABLE

bool tflite_inference_begin() {
    Serial.println("# tflite_inference: TFLM library not installed -- see comment "
                    "at the top of tflite_inference.cpp. Inference disabled.");
    return false;
}

AnnOutput tflite_infer(const AnnInput& /*in*/) {
    AnnOutput out{};
    out.valid = false;
    return out;
}

#endif

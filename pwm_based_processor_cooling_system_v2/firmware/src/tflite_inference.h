#pragma once
#include <Arduino.h>

// TensorFlow Lite Micro inference -- THIS build calls it for real, every
// control tick, in ANN control mode (see control_modes.h). The model comes
// from python/convert_tflite.py, trained offline on the BMCDATA dataset --
// no physical data-acquisition firmware exists in this project; see
// README.md.

struct AnnInput {
    float proc_temp_c;    // avg(DS18B20 #1, DS18B20 #2) on this hardware,
                           // standing in for avg(Cpu1_Temp, Cpu2_Temp) in training
    float dTdt_c_per_s;
    float inlet_temp_c;    // ambient DS18B20, standing in for BMCDATA's Inlet_Temp
    float cpu_usage;       // 0-100, standing in for BMCDATA's real cpu_usage --
                            // this hardware has no real CPU to read utilization
                            // from, so workload.cpp's busy-fraction*100 is used
                            // as a proxy (see main.cpp and the README's honesty
                            // note about what this validates vs. doesn't)
};

struct AnnOutput {
    float pwm_pct;         // rescaled from the model's raw 0-1 pwm_frac output --
                            // see python/model.py's docstring for why the model
                            // itself outputs a fraction, not a percentage
    float future_temp_c;
    float thermal_risk;
    bool  valid;
};

// Allocates the tensor arena and loads the embedded model (model_data.h).
// Returns false if the model failed AllocateTensors (arena too small, etc).
bool tflite_inference_begin();

// Applies the same StandardScaler mean/scale used in training (see
// models/scaler_params.json, hard-coded into tflite_inference.cpp after
// training -- do not forget to update those constants when you retrain).
AnnOutput tflite_infer(const AnnInput& in);

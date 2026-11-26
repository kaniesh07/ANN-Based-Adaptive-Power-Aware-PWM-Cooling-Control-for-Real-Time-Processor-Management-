// ANN-Based Adaptive Power-Aware PWM Cooling Control -- Deployment Firmware
//
// This project trains its ANN offline on the public BMCDATA dataset (see
// python/), NOT on data acquired by this board -- there is no acquisition
// sweep in this firmware. What's running here is real-time inference +
// control, plus the Fixed-PWM and Rule-based baseline modes the project's
// own evaluation section calls for, so the same hardware can produce all
// three comparison runs (switch ACTIVE_CONTROL_MODE in config.h, reflash,
// run a validation session, repeat).
//
// Every control tick: read DS18B20 (x2) + INA219, compute dT/dt, build the
// 4-feature ANN input, run inference (or the active baseline controller),
// apply PWM, log one CSV row.
//
// HONESTY NOTE (see README for the full version): this ESP32 has no real
// CPU workload to read utilization from the way BMCDATA's host-side
// cpu_usage column does -- workload.cpp's busy-loop duty fraction is used
// as a workload proxy. This firmware validates the CONTROL LOGIC the ANN
// learned from real server thermal data, not a live re-measurement of
// server-scale thermals on ESP32-scale hardware.

#include <Arduino.h>
#include "config.h"
#include "sensors.h"
#include "fan_control.h"
#include "workload.h"
#include "tflite_inference.h"
#include "rule_based_control.h"
#include "logger.h"

static float    g_prev_proc_temp_c = NAN;
static uint32_t g_prev_sample_ms   = 0;
static bool     g_have_prev        = false;

void setup() {
    Serial.begin(115200);
    delay(500);
    pinMode(PIN_STATUS_LED, OUTPUT);

#if DISCOVER_ADDRESSES
    Serial.println("# DISCOVER_ADDRESSES mode -- not running control.");
    sensors_begin();
    sensors_discover_addresses();
    while (true) { delay(1000); }
#endif

    bool sensors_ok = sensors_begin();
    fan_control_begin();
#if FAN_TEST_ON_BOOT
    fan_self_test();
#endif

    // Loaded regardless of control mode -- ANN predictions are logged as a
    // monitoring signal even when a baseline controller is driving the fan.
    bool ann_ok = tflite_inference_begin();

    Serial.println("# ANN Cooling Control -- Deployment Firmware");
    Serial.print("# control_mode = ");
    Serial.println(ACTIVE_CONTROL_MODE == CONTROL_ANN ? "ANN" :
                    ACTIVE_CONTROL_MODE == CONTROL_FIXED_PWM ? "FIXED_PWM" : "RULE_BASED");
    if (!sensors_ok) {
        Serial.println("# NOTE: config.h DS18B20 addresses are placeholders until you run");
        Serial.println("#       DISCOVER_ADDRESSES=1 once and paste the real ROM codes in.");
        Serial.println("#       Until then sensor_valid=0 and the fan sits at FAN_MIN_RUN_DUTY.");
    }
    Serial.print("# sensors_begin() -> ");
    Serial.println(sensors_ok ? "OK" : "CHECK WIRING (see warnings above)");
    Serial.print("# tflite_inference_begin() -> ");
    Serial.println(ann_ok ? "OK" : "FAILED (see warning above) -- ann_valid will read 0 in every logged row");
    logger_print_header();
}

void loop() {
    digitalWrite(PIN_STATUS_LED, (millis() / 1000) % 2);

    // Generate this tick's workload proxy (blocks ~CONTROL_TICK_MS) and
    // derive the 0-100 cpu_usage-equivalent input the model expects.
    workload_run_tick(WORKLOAD_LEVEL);
    float cpu_usage_proxy = WORKLOAD_BUSY_FRACTION[WORKLOAD_LEVEL] * 100.0f;

    SensorReading reading = sensors_read();
    uint32_t now_ms = millis();

    float dTdt = 0.0f;
    if (g_have_prev && reading.valid) {
        uint32_t dt_ms = now_ms - g_prev_sample_ms;
        if (dt_ms >= DTDT_MIN_INTERVAL_MS) {
            dTdt = (reading.proc_temp_c - g_prev_proc_temp_c) / (dt_ms / 1000.0f);
        }
    }
    if (reading.valid) {
        g_prev_proc_temp_c = reading.proc_temp_c;
        g_prev_sample_ms = now_ms;
        g_have_prev = true;
    }

    // Run ANN inference every tick regardless of control mode -- in
    // non-ANN modes its PWM output is ignored for actuation, but
    // future_temp_c/thermal_risk are still logged as a monitoring signal
    // alongside whichever controller is actually driving the fan (those
    // two outputs are the ones worth trusting right now; see config.h).
    AnnInput ann_in{};
    ann_in.proc_temp_c   = reading.proc_temp_c;
    ann_in.dTdt_c_per_s  = dTdt;
    ann_in.inlet_temp_c  = reading.ambient_temp_c;
    ann_in.cpu_usage     = cpu_usage_proxy;
    AnnOutput ann_out = tflite_infer(ann_in);

    // --- Decide PWM for this tick, per the active control mode ---
    uint8_t duty = 0;
    switch (ACTIVE_CONTROL_MODE) {
        case CONTROL_ANN:
            if (ann_out.valid && reading.valid) {
                float pct = constrain(ann_out.pwm_pct, 0.0f, 100.0f);
                duty = (uint8_t)(pct / 100.0f * 255.0f);
                // FIX: the INT8 PWM head reads exactly 0 below ~50 C, which
                // would switch the fan off. Never let the ANN stop the fan.
                if (duty < FAN_MIN_RUN_DUTY) duty = FAN_MIN_RUN_DUTY;
            } else {
                // No usable inference this tick (model not loaded, or bad
                // sensor read) -- fail toward safety, not toward silence.
                duty = reading.valid && reading.proc_temp_c >= THERMAL_LIMIT_C ? 255 : RULE_PWM_MIN;
            }
            break;
        case CONTROL_FIXED_PWM:
            duty = FIXED_PWM_BASELINE_DUTY;
            break;
        case CONTROL_RULE_BASED:
            duty = reading.valid ? rule_based_pwm(reading.proc_temp_c) : RULE_PWM_MIN;
            break;
    }

    // Hard safety override, regardless of control mode -- logged as the
    // commanded value from the controller above, then overridden before
    // being applied, same convention as the acquisition-sweep version of
    // this project used.
    if (reading.valid && reading.proc_temp_c >= THERMAL_LIMIT_C) {
        duty = 255;
        Serial.println("# SAFETY: proc_temp_c >= THERMAL_LIMIT_C, forcing PWM=255");
    }

    fan_set_pwm(duty);

    logger_print_row(now_ms, ACTIVE_CONTROL_MODE, reading, dTdt, cpu_usage_proxy, duty, ann_out);
}

#pragma once
#include <Arduino.h>
#include "config.h"
#include "sensors.h"
#include "tflite_inference.h"

void logger_print_header();

void logger_print_row(
    uint32_t timestamp_ms,
    ControlMode mode,
    const SensorReading& reading,
    float dTdt,
    float cpu_usage_proxy,
    uint8_t applied_pwm_duty,
    const AnnOutput& ann_out   // valid=false rows just log 0s for the ANN-only fields
);

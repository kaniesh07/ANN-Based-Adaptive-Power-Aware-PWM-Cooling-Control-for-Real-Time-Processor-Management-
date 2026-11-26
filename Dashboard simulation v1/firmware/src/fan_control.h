#pragma once
#include <Arduino.h>

void fan_control_begin();

// duty: 0-255
void fan_set_pwm(uint8_t duty);

// Returns duty as a 0-100 percentage (for logging).
float fan_get_pwm_percent();

// Boot-time wiring check: steps the raw duty 0 -> 100% and prints each step.
void fan_self_test();

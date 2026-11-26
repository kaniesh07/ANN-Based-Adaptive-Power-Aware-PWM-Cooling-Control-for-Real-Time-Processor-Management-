#pragma once
#include <Arduino.h>

// Linearly ramps PWM between RULE_PWM_MIN and RULE_PWM_MAX as proc_temp_c
// goes from RULE_TEMP_LOW_C to RULE_TEMP_HIGH_C (config.h). This is the
// "conventional" baseline the project's evaluation section (11) compares
// the ANN against -- deliberately simple (temperature-only, no dT/dt, no
// workload awareness) so the comparison actually demonstrates what the ANN
// adds, rather than being a strawman or a disguised second ANN.
uint8_t rule_based_pwm(float proc_temp_c);

#include "rule_based_control.h"
#include "config.h"

uint8_t rule_based_pwm(float proc_temp_c) {
    if (proc_temp_c <= RULE_TEMP_LOW_C) return RULE_PWM_MIN;
    if (proc_temp_c >= RULE_TEMP_HIGH_C) return RULE_PWM_MAX;

    float frac = (proc_temp_c - RULE_TEMP_LOW_C) / (RULE_TEMP_HIGH_C - RULE_TEMP_LOW_C);
    return (uint8_t)(RULE_PWM_MIN + frac * (RULE_PWM_MAX - RULE_PWM_MIN));
}

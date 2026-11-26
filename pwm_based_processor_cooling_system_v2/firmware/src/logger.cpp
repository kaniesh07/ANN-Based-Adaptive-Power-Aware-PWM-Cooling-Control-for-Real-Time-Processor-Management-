#include "logger.h"

static const char* mode_to_string(ControlMode m) {
    switch (m) {
        case CONTROL_ANN:        return "ANN";
        case CONTROL_FIXED_PWM:  return "FIXED_PWM";
        case CONTROL_RULE_BASED: return "RULE_BASED";
        default:                 return "UNKNOWN";
    }
}

void logger_print_header() {
    Serial.println(
        "#DATA,timestamp_ms,control_mode,proc_temp_c,inlet_temp_c,"
        "dTdt_c_per_s,cpu_usage_proxy,applied_pwm_pct,"
        "ann_pwm_pct,ann_future_temp_c,ann_thermal_risk,ann_valid,"
        "fan_voltage_v,fan_current_ma,fan_power_mw,sensor_valid"
    );
}

void logger_print_row(
    uint32_t timestamp_ms,
    ControlMode mode,
    const SensorReading& reading,
    float dTdt,
    float cpu_usage_proxy,
    uint8_t applied_pwm_duty,
    const AnnOutput& ann_out
) {
    float applied_pwm_pct = (applied_pwm_duty / 255.0f) * 100.0f;

    Serial.printf(
        "DATA,%lu,%s,%.3f,%.3f,%.4f,%.2f,%.2f,%.2f,%.3f,%.4f,%d,%.3f,%.2f,%.2f,%d\n",
        (unsigned long)timestamp_ms,
        mode_to_string(mode),
        reading.proc_temp_c,
        reading.ambient_temp_c,  // this is the "inlet" sensor physically, see sensors.h
        dTdt,
        cpu_usage_proxy,
        applied_pwm_pct,
        ann_out.pwm_pct,
        ann_out.future_temp_c,
        ann_out.thermal_risk,
        ann_out.valid ? 1 : 0,
        reading.fan_voltage_v,
        reading.fan_current_ma,
        reading.fan_power_mw,
        reading.valid ? 1 : 0
    );
}

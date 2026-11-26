#include "fan_control.h"
#include "config.h"

static uint8_t g_current_duty = 0;

static void pwm_write_raw(uint8_t duty) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
    ledcWrite(PIN_FAN_GATE, duty);
#else
    ledcWrite(PWM_CHANNEL, duty);
#endif
}

void fan_control_begin() {
    // Hold the gate low while LEDC is being set up so the fan doesn't
    // twitch on a floating pin at reset. (Also fit a 10k gate->GND
    // pull-down on the board -- see FAN_FIX_NOTES.md.)
    pinMode(PIN_FAN_GATE, OUTPUT);
    digitalWrite(PIN_FAN_GATE, LOW);

#if ESP_ARDUINO_VERSION_MAJOR >= 3
    // Arduino-ESP32 core 3.x LEDC API: attach directly to the pin.
    ledcAttach(PIN_FAN_GATE, PWM_FREQ_HZ, PWM_RESOLUTION);
#else
    // Arduino-ESP32 core 2.x LEDC API: setup a channel, then attach the pin.
    ledcSetup(PWM_CHANNEL, PWM_FREQ_HZ, PWM_RESOLUTION);
    ledcAttachPin(PIN_FAN_GATE, PWM_CHANNEL);
#endif
    pwm_write_raw(0);
    g_current_duty = 0;
}

void fan_set_pwm(uint8_t duty) {
    // FIX 1: minimum running duty. A non-zero command below the stall point
    // just hums the motor without turning it, so raise it to the floor.
    if (duty > 0 && duty < FAN_MIN_RUN_DUTY) {
        duty = FAN_MIN_RUN_DUTY;
    }

    // FIX 2: kick-start. Starting torque needs much more than running torque,
    // so going from stopped to running gets a short full-power pulse first.
    if (g_current_duty == 0 && duty > 0) {
        pwm_write_raw(FAN_KICK_DUTY);
        delay(FAN_KICK_MS);
    }

    g_current_duty = duty;
    pwm_write_raw(duty);
}

float fan_get_pwm_percent() {
    return (g_current_duty / 255.0f) * 100.0f;
}

void fan_self_test() {
    Serial.println("# FAN SELF-TEST: sweeping duty, 3 s per step. Watch the fan.");
    Serial.println("# If it never turns, even at 100%, the problem is wiring/power, not code.");
    const uint8_t steps[] = {0, 51, 77, 102, 128, 179, 255, 0};
    for (uint8_t d : steps) {
        pwm_write_raw(d);   // raw on purpose: no floor, no kick, so you see the true stall point
        Serial.printf("# FAN SELF-TEST: duty=%3u/255 (%5.1f%%)\n", d, d / 2.55f);
        delay(3000);
    }
    g_current_duty = 0;
    Serial.println("# FAN SELF-TEST done. Set FAN_MIN_RUN_DUTY just above the lowest step that kept it turning.");
}

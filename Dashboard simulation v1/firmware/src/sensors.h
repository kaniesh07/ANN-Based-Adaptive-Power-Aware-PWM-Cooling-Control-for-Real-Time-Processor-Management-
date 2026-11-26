#pragma once
#include <Arduino.h>

struct SensorReading {
    float proc_temp_c;
    float ambient_temp_c;
    float fan_voltage_v;
    float fan_current_ma;
    float fan_power_mw;
    bool  valid;
};

// Call once from setup()
bool sensors_begin();

// Blocking-ish read of both DS18B20s + INA219. Handles the 750ms conversion
// wait internally (call this from the main control tick, not in a tight loop).
SensorReading sensors_read();

// Runs address discovery: prints every DS18B20 ROM address found on the
// OneWire bus to Serial. Only used when config.h DISCOVER_ADDRESSES == 1.
void sensors_discover_addresses();

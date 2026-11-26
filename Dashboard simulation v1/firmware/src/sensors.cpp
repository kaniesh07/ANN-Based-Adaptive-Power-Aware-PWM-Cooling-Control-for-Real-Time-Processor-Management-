#include "sensors.h"
#include "config.h"
#include <OneWire.h>
#include <DallasTemperature.h>
#if USE_INA219
#include <Wire.h>
#include <Adafruit_INA219.h>
static Adafruit_INA219 ina219;
#endif

static OneWire oneWire(PIN_ONEWIRE_BUS);
static DallasTemperature ds18(&oneWire);

bool sensors_begin() {
    ds18.begin();
    ds18.setResolution(12);
    ds18.setWaitForConversion(false); // we manage the wait ourselves

    bool ina_ok = true;
#if USE_INA219
    Wire.begin();
    ina_ok = ina219.begin();
    if (!ina_ok) {
        Serial.println("# WARNING: INA219 not found at default I2C address (0x40).");
    }
#endif

    uint8_t count = ds18.getDeviceCount();
    if (count < 2) {
        Serial.print("# WARNING: expected 2 DS18B20 sensors on the bus, found ");
        Serial.println(count);
    }
    return ina_ok && (count >= 2);
}

void sensors_discover_addresses() {
    Serial.println("# Scanning OneWire bus for DS18B20 ROM addresses...");
    oneWire.reset_search();
    DeviceAddress addr;
    int found = 0;
    while (oneWire.search(addr)) {
        found++;
        Serial.printf("# Device %d: {", found);
        for (uint8_t i = 0; i < 8; i++) {
            Serial.printf("0x%02X%s", addr[i], (i < 7) ? ", " : "");
        }
        Serial.println("}");
    }
    if (found == 0) {
        Serial.println("# No devices found -- check wiring / 4.7k pull-up on data line.");
    }
    Serial.println("# Copy each address into ADDR_PROCESSOR_SENSOR / ADDR_AMBIENT_SENSOR");
    Serial.println("# in config.h, decide which physical sensor is which role, then set");
    Serial.println("# DISCOVER_ADDRESSES back to 0 and reflash.");
}

SensorReading sensors_read() {
    SensorReading r{};
    r.valid = true;

    ds18.requestTemperaturesByAddress(ADDR_PROCESSOR_SENSOR);
    ds18.requestTemperaturesByAddress(ADDR_AMBIENT_SENSOR);
    // Both conversions were kicked off close together; a single wait covers both.
    delay(DS18B20_CONVERSION_MS);

    r.proc_temp_c    = ds18.getTempC(ADDR_PROCESSOR_SENSOR);
    r.ambient_temp_c = ds18.getTempC(ADDR_AMBIENT_SENSOR);

    if (r.proc_temp_c == DEVICE_DISCONNECTED_C || r.ambient_temp_c == DEVICE_DISCONNECTED_C) {
        r.valid = false;
    }

#if USE_INA219
    r.fan_voltage_v  = ina219.getBusVoltage_V() + (ina219.getShuntVoltage_mV() / 1000.0f);
    r.fan_current_ma = ina219.getCurrent_mA();
    r.fan_power_mw   = ina219.getPower_mW();
#else
    // No INA219 fitted: fan power fields log as 0. Measure fan V/I manually
    // at each PWM level instead (see README).
    r.fan_voltage_v = r.fan_current_ma = r.fan_power_mw = 0.0f;
#endif

    return r;
}

#pragma once
#include <Arduino.h>

// =====================================================================
// PIN MAP (unchanged from the physical-acquisition version of this
// project -- same prototype hardware, now running inference instead of
// an acquisition sweep)
// =====================================================================
static const uint8_t PIN_ONEWIRE_BUS   = 4;
static const uint8_t PIN_FAN_GATE      = 25;
static const uint8_t PIN_STATUS_LED    = 2;

#define DISCOVER_ADDRESSES 0

// INA219 power monitor. 0 = not fitted (no I2C access, fan V/I/P log as 0).
// Set to 1 only if you wire the INA219 back in.
#define USE_INA219 0

static uint8_t ADDR_PROCESSOR_SENSOR[8] = {0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
static uint8_t ADDR_AMBIENT_SENSOR[8]   = {0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01};

static const uint8_t  PWM_CHANNEL    = 0;
static const uint32_t PWM_FREQ_HZ    = 20000;
static const uint8_t  PWM_RESOLUTION = 8;

// =====================================================================
// CONTROL MODE -- the evaluation this project calls for (section 11:
// ANN vs Fixed-PWM vs Rule-based) means the SAME hardware needs to run
// under three different controllers for comparison. Pick one, flash,
// run a validation session, repeat for the other two -- rather than
// three separate firmware projects.
// =====================================================================
enum ControlMode : uint8_t {
    CONTROL_ANN        = 0,  // tflite_inference.cpp -- the actual project output
    CONTROL_FIXED_PWM  = 1,  // constant PWM regardless of temperature -- baseline
    CONTROL_RULE_BASED = 2   // simple threshold curve -- see rule_based_control.cpp
};

// Trained on the full 122-file BMCDATA dataset -- the ANN's future_temp_c
// and thermal_risk outputs are reliable (test MAE 0.57C and 0.05 resp.),
// but its PWM output is NOT (test MAE ~12 percentage points, see README's
// "Honest results" section for why -- it's a data-sparsity problem in the
// dataset itself, not a bug). CONTROL_ANN is left as the default because
// it's this project's actual deliverable, but for anything you'd actually
// trust unattended, CONTROL_RULE_BASED is the safer choice today -- it
// still logs the ANN's future_temp/risk predictions alongside its own PWM
// decision (see main.cpp), so you get the ANN's reliable outputs as a
// monitoring signal without trusting its PWM choice.
// FIX: switched to RULE_BASED so the fan actually runs. The INT8 model's
// PWM output is quantized in steps of ~22% and reads exactly 0% for any
// proc_temp below ~50 C (see FAN_FIX_NOTES.md), so CONTROL_ANN turns the fan
// OFF at room temperature. ANN future_temp/risk are still logged every tick.
static const ControlMode ACTIVE_CONTROL_MODE = CONTROL_RULE_BASED;
static const uint8_t FIXED_PWM_BASELINE_DUTY = 128;  // only used in CONTROL_FIXED_PWM, 0-255

// =====================================================================
// WORKLOAD PROXY
// This hardware has no real CPU to read utilization from the way the
// BMCDATA dataset's host-side cpu_usage column does. workload.cpp's
// busy-fraction CPU loop stands in for it, and WORKLOAD_LEVEL below
// picks which fixed level runs for the whole session -- there's no
// acquisition sweep here, this is steady-state validation, not data
// collection. See README for what this does and doesn't validate.
// =====================================================================
enum WorkloadLevel : uint8_t { WORKLOAD_LIGHT = 0, WORKLOAD_MEDIUM = 1, WORKLOAD_HEAVY = 2 };
static const WorkloadLevel WORKLOAD_LEVEL = WORKLOAD_MEDIUM;

static const uint16_t CONTROL_TICK_MS = 1000;
static const float WORKLOAD_BUSY_FRACTION[3] = {0.20f, 0.55f, 0.90f};  // -> cpu_usage proxy, 0-100

// =====================================================================
// THERMAL SAMPLING / SAFETY
// =====================================================================
static const uint16_t DS18B20_CONVERSION_MS = 750;
static const uint32_t DTDT_MIN_INTERVAL_MS  = 900;
static const float    THERMAL_LIMIT_C       = 85.0f;  // MUST match python/data_pipeline.py's
                                                        // THERMAL_LIMIT_C -- the ANN was trained
                                                        // against this value; changing one without
                                                        // the other makes the risk score meaningless

// =====================================================================
// RULE-BASED BASELINE CONTROLLER (see rule_based_control.cpp)
// =====================================================================
static const float RULE_TEMP_LOW_C   = 45.0f;  // below this: minimum PWM
static const float RULE_TEMP_HIGH_C  = 75.0f;  // at/above this: full PWM
// =====================================================================
// FAN DRIVE (FIX) -- most small DC fans will not START below ~30-40% duty
// when the supply is chopped by a low-side MOSFET, and some stall below
// ~25% once running. 20% (the old RULE_PWM_MIN) leaves the fan dead still.
//   FAN_MIN_RUN_DUTY : any non-zero command is raised to at least this
//   FAN_KICK_DUTY/MS : when going from stopped -> running, give a full-power
//                      pulse first so the rotor breaks free, then drop down
//   FAN_TEST_ON_BOOT : 1 = run a 0->100% sweep at boot, printing each step,
//                      so you can check the wiring before any control runs
// Tune FAN_MIN_RUN_DUTY with the boot sweep: set it a bit above the lowest
// step where your fan keeps turning.
// =====================================================================
static const uint8_t  FAN_MIN_RUN_DUTY = 102;   // 40%
static const uint8_t  FAN_KICK_DUTY    = 255;   // 100%
static const uint16_t FAN_KICK_MS      = 600;
#define FAN_TEST_ON_BOOT 1

static const uint8_t RULE_PWM_MIN    = FAN_MIN_RUN_DUTY;  // was 51 (20%) -- too low to spin most fans
static const uint8_t RULE_PWM_MAX    = 255;   // 100%

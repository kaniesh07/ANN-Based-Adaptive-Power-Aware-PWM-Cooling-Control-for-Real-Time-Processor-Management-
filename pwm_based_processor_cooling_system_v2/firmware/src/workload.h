#pragma once
#include <Arduino.h>
#include "config.h"

// Busy-computes for (WORKLOAD_BUSY_FRACTION[level] * CONTROL_TICK_MS), then
// returns. The caller is expected to call this once per control tick so the
// tick period stays close to CONTROL_TICK_MS regardless of workload level.
void workload_run_tick(WorkloadLevel level);

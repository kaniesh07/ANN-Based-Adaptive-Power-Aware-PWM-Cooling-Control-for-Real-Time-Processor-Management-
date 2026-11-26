#include "workload.h"

// Deliberately expensive, hard-to-optimize-away math so the compiler can't
// just fold the busy-loop away at -O2. volatile sink forces each iteration
// to actually execute.
static volatile float g_sink = 0.0f;

static void burn_cpu_for(uint32_t ms) {
    uint32_t start = millis();
    float x = 1.0001f;
    while ((millis() - start) < ms) {
        for (int i = 0; i < 2000; i++) {
            x = sinf(x) * cosf(x) + sqrtf(fabsf(x)) + 1.0f;
            if (x > 1e6f || x < -1e6f) x = 1.0001f; // keep it bounded
        }
    }
    g_sink = x;
}

void workload_run_tick(WorkloadLevel level) {
    uint32_t busy_ms = (uint32_t)(WORKLOAD_BUSY_FRACTION[level] * CONTROL_TICK_MS);
    uint32_t idle_ms = CONTROL_TICK_MS - busy_ms;

    burn_cpu_for(busy_ms);
    if (idle_ms > 0) {
        delay(idle_ms);
    }
}

# Why the fan doesn't run, and what was changed

## Firmware causes (found in the code)

| # | Cause | Effect | Fix in this version |
|---|---|---|---|
| 1 | `ACTIVE_CONTROL_MODE = CONTROL_ANN`, and the INT8 model's 3 outputs share ONE quantization scale (0.222) because `model.py` concatenates them into a single tensor | PWM head can only be 0%, 22%, 44%... and reads **exactly 0%** for any proc temp below ~50 °C. At room temperature the ANN commands the fan OFF. (Checked by running `cooling_ann_int8.tflite`: 28-45 °C -> 0%, 55-75 °C -> 22%.) | Default mode is now `CONTROL_RULE_BASED`; in ANN mode the duty is floored at `FAN_MIN_RUN_DUTY` |
| 2 | TFLite Micro library not in `platformio.ini` | ANN compiles out, `ann_valid=0`, fallback duty = 51 (20%) | Not needed for RULE_BASED; commented line added to enable later |
| 3 | DS18B20 addresses in `config.h` are placeholders (`0x28,0,0,...`) | Every read returns -127 °C, `sensor_valid=0`, fallback duty = 51 (20%) | Boot message tells you; run `DISCOVER_ADDRESSES 1` once and paste real ROM codes |
| 4 | Minimum duty 51/255 = 20%, no kick-start | Most small DC fans need ~30-40% to **start** from rest when the supply is chopped -> they hum but don't turn | `FAN_MIN_RUN_DUTY = 102` (40%) + 600 ms full-power kick when starting from 0 |
| 5 | Input quantization used a bare `(int8_t)` cast | Truncates, and wraps around for out-of-range inputs (-127 °C becomes a large positive value) | Rounded and clamped to -128..127 |

New: `FAN_TEST_ON_BOOT 1` sweeps the raw duty 0 -> 20 -> 30 -> 40 -> 50 -> 70 -> 100% (3 s each) at boot and prints each step on Serial. That tells you instantly whether the problem is code or wiring.

## Hardware checklist (IRLZ44N low-side switch)

```
 Fan supply +V (5 V or 12 V) ──┬─────────────┐
                               │             │
                         [Fan +]        Diode cathode (1N5819 / 1N4007)
                         [Fan −]        Diode anode
                               │             │
                               └──── Drain ──┘
                                     IRLZ44N
 GPIO25 ──[ 220 Ω ]──── Gate          Source
                         │              │
                      [ 10 kΩ ]         │
                         │              │
 ESP32 GND ──────────────┴──────────────┴──── Fan supply GND   <- MUST be common
```

1. **Common ground.** ESP32 GND and the fan supply's GND must be joined. This is the #1 reason a MOSFET fan driver "does nothing".
2. **Pin order.** IRLZ44N (TO-220, tab facing away, legs down): **G - D - S** left to right. Swapping D and S puts the body diode in circuit and the fan runs always-on or never.
3. **Gate resistor 100-220 Ω and 10 kΩ gate-to-source pull-down.** The pull-down keeps the fan off during boot.
4. **Flyback diode across the fan**, cathode to +V. Without it the switching spikes can damage the MOSFET.
5. **Capacitors:** do NOT put a capacitor from Drain to GND -- the MOSFET shorts it 20,000 times a second. Put a bulk cap (100-470 µF) across the fan supply rails (+V to GND) instead.
6. **3.3 V gate drive.** IRLZ44N's on-resistance is only specified at 4-5 V gate. At 3.3 V it is fine for a small fan (< ~0.3 A). For a bigger fan, use a MOSFET rated at 2.5 V gate (AO3400, IRLB8721) or add an NPN/driver stage.
7. **4-wire PC fan?** Leave the blue PWM wire unconnected (= full speed) while testing this circuit. The cleaner long-term option is constant 12 V to the fan and GPIO25 driving the blue wire through an NPN/open-drain at 25 kHz -- then the MOSFET is not needed.
8. **Fan clicks or stutters at low duty?** Some brushless fans' internal controller dislikes a chopped supply. Try `PWM_FREQ_HZ = 100` or raise `FAN_MIN_RUN_DUTY`.

## 2-minute bench test

1. Power everything, ESP32 not flashed or held in reset. Put a wire from **Drain to GND**. The fan must spin. If it doesn't: fan supply or fan problem, not the circuit.
2. Remove the wire. Connect **Gate to 3.3 V** directly. The fan must spin; Drain-to-Source voltage should be < 0.3 V. If not: MOSFET pins, ground, or a dead MOSFET.
3. Flash this firmware, open Serial Monitor at 115200, watch the boot sweep. Note the lowest step where the fan keeps turning and set `FAN_MIN_RUN_DUTY` a little above it.
4. Set `DISCOVER_ADDRESSES 1`, flash, copy the two printed ROM codes into `config.h`, set it back to `0`, flash again. `sensor_valid` should now read 1.

## Proper fix for the ANN's PWM output (needs retraining/re-export)

In `python/model.py`, return the three heads as **separate outputs** instead of `layers.Concatenate`, so each gets its own INT8 scale (PWM 0-1 gets ~0.004 resolution instead of 0.222; future_temp stops saturating at 56.6 °C). Update `tflite_inference.cpp` to read `output(0)`, `output(1)`, `output(2)` with each one's own `scale`/`zero_point`. Even then, the README's caveat stands: the dataset only has 4 distinct PWM levels, so keep RULE_BASED driving the fan.

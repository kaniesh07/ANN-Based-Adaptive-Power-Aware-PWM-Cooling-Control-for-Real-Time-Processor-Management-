# Cooling dashboard (Streamlit)

Live view of the ESP32 log + controller comparison + model-performance plots.

## Run it

```
cd pwm_cooling_project_v2/dashboard
pip install -r requirements.txt
streamlit run app.py
```
It opens in your browser at http://localhost:8501.

## Tabs

- **Live monitor** - sidebar -> Source:
  - *Demo (simulated)*: try the dashboard without the board.
  - *ESP32 serial port*: pick the COM port (115200 baud). **Close the PlatformIO
    Serial Monitor first** - only one program can hold the port.
    Every run is saved automatically to `dashboard/logs/run_<date>.csv`.
  - *Open a saved log*: a `logs/run_*.csv` or a pasted Serial Monitor capture.
- **Controller comparison** - upload one log per controller (FIXED_PWM,
  RULE_BASED, ANN). Shows mean/peak temperature, mean PWM, fan energy, and
  overlaid traces. Without the INA219, energy is estimated from the
  PWM -> power table (put your multimeter readings in it).
  `sample_logs/` has three simulated runs so you can see how it looks.
- **Model performance** - the training plots from `../plots` and the test MAE table.

## Typical workflow

1. Flash the firmware with `ACTIVE_CONTROL_MODE = CONTROL_FIXED_PWM`, start the
   dashboard on the serial port, run ~15 min. Repeat for RULE_BASED and ANN,
   same workload level and starting temperature.
2. Open the Comparison tab and upload the three files from `logs/`.

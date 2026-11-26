"""
PWM Cooling Control -- Streamlit dashboard

Three tabs:
  Live monitor        -- reads the ESP32's "DATA,..." serial log in real time
                         (or a built-in simulator when no board is plugged in)
                         and plots temperature, fan PWM and the ANN's predictions.
  Controller comparison -- load the CSV logs from your FIXED_PWM / RULE_BASED /
                         ANN runs and compare temperature, PWM and fan energy.
  Model performance   -- the offline training plots from ../plots and the
                         test-set metrics from the README.

Run:   streamlit run app.py
"""
from __future__ import annotations

import io
import math
import random
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import serial
    import serial.tools.list_ports
    HAVE_SERIAL = True
except ImportError:  # dashboard still works for CSV / demo without pyserial
    HAVE_SERIAL = False

# ---------------------------------------------------------------------------
# Constants shared with the firmware
# ---------------------------------------------------------------------------
COLUMNS = [
    "timestamp_ms", "control_mode", "proc_temp_c", "inlet_temp_c",
    "dTdt_c_per_s", "cpu_usage_proxy", "applied_pwm_pct",
    "ann_pwm_pct", "ann_future_temp_c", "ann_thermal_risk", "ann_valid",
    "fan_voltage_v", "fan_current_ma", "fan_power_mw", "sensor_valid",
]
NUMERIC = [c for c in COLUMNS if c != "control_mode"]
THERMAL_LIMIT_C = 85.0      # config.h THERMAL_LIMIT_C
RULE_TEMP_LOW_C = 45.0      # config.h RULE_TEMP_LOW_C
RULE_TEMP_HIGH_C = 75.0     # config.h RULE_TEMP_HIGH_C

HERE = Path(__file__).resolve().parent
PLOTS_DIR = HERE.parent / "plots"
LOG_DIR = HERE / "logs"

# Categorical colours, fixed order (validated palette, first three slots are
# safe for colour-blind readers even when all three appear together).
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_MUTED = "#8a8984"
MODE_COLORS = {"FIXED_PWM": C_BLUE, "RULE_BASED": C_ORANGE, "ANN": C_AQUA}

st.set_page_config(page_title="PWM Cooling Dashboard", page_icon="🌀", layout="wide")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_data_line(line: str) -> dict | None:
    """One 'DATA,...' line from logger.cpp -> dict, or None if not a data row."""
    line = line.strip()
    if not line.startswith("DATA,"):
        return None
    parts = line[5:].split(",")
    if len(parts) != len(COLUMNS):
        return None
    row = dict(zip(COLUMNS, parts))
    for c in NUMERIC:
        try:
            row[c] = float(row[c])
        except ValueError:
            row[c] = float("nan")
    return row


def parse_log_text(text: str) -> pd.DataFrame:
    """Accepts a raw Serial Monitor capture (DATA, lines mixed with # lines)
    or an already-clean CSV with a header row."""
    rows = [r for r in (parse_data_line(l) for l in text.splitlines()) if r]
    if rows:
        return tidy(pd.DataFrame(rows))
    try:
        df = pd.read_csv(io.StringIO(text), comment="#")
        if {"timestamp_ms", "proc_temp_c", "applied_pwm_pct"} <= set(df.columns):
            return tidy(df)
    except Exception:
        pass
    return pd.DataFrame(columns=COLUMNS)


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Invalid sensor reads come through as -127 C -- blank them so they don't
    # wreck the temperature axis.
    if "sensor_valid" in df:
        bad = df["sensor_valid"] < 0.5
        df.loc[bad, ["proc_temp_c", "inlet_temp_c", "dTdt_c_per_s"]] = np.nan
    if "ann_valid" in df:
        bad = df["ann_valid"] < 0.5
        df.loc[bad, ["ann_pwm_pct", "ann_future_temp_c", "ann_thermal_risk"]] = np.nan
    t0 = df["timestamp_ms"].iloc[0]
    df["t_s"] = (df["timestamp_ms"] - t0) / 1000.0
    return df


# ---------------------------------------------------------------------------
# Live sources (one background thread per source, shared across reruns)
# ---------------------------------------------------------------------------
class LiveSource:
    """Background reader. Keeps the last N rows + the last few '#' messages,
    and appends every raw line to logs/<timestamp>.csv so each run is saved."""

    def __init__(self, max_rows: int = 3600):
        self.rows: deque = deque(maxlen=max_rows)
        self.messages: deque = deque(maxlen=40)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.error: str | None = None
        LOG_DIR.mkdir(exist_ok=True)
        self.log_path = LOG_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self._log = open(self.log_path, "w", encoding="utf-8")
        self._log.write("#DATA," + ",".join(COLUMNS) + "\n")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _handle(self, line: str):
        row = parse_data_line(line)
        with self.lock:
            if row:
                self.rows.append(row)
            elif line.startswith("#"):
                self.messages.append(line.strip())
        if row or line.startswith("#"):
            self._log.write(line.strip() + "\n")
            self._log.flush()

    def snapshot(self) -> tuple[pd.DataFrame, list[str]]:
        with self.lock:
            rows, msgs = list(self.rows), list(self.messages)
        return (tidy(pd.DataFrame(rows)) if rows else pd.DataFrame(columns=COLUMNS)), msgs

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        try:
            self._log.close()
        except Exception:
            pass

    def _run(self):
        raise NotImplementedError


class SerialSource(LiveSource):
    def __init__(self, port: str, baud: int):
        self.port, self.baud = port, baud
        super().__init__()

    def _run(self):
        try:
            with serial.Serial(self.port, self.baud, timeout=1) as ser:
                while not self.stop_event.is_set():
                    raw = ser.readline()
                    if raw:
                        self._handle(raw.decode("utf-8", errors="replace"))
        except Exception as e:  # port busy / unplugged
            self.error = f"{type(e).__name__}: {e}"


class DemoSource(LiveSource):
    """Simulates the firmware so the dashboard can be tried without a board:
    a heated block warming under load, the rule-based fan curve, and an ANN
    that predicts ~10 s ahead."""

    def __init__(self, mode: str = "RULE_BASED"):
        self.mode = mode
        super().__init__()

    def _run(self):
        T, Tin, prev = 32.0, 27.0, 32.0
        t0 = time.time()
        self._handle("# DEMO MODE -- simulated data, no board connected")
        while not self.stop_event.is_set():
            t = time.time() - t0
            load = 55 + 35 * (math.sin(t / 40) > 0.3)          # medium / heavy bursts
            if self.mode == "RULE_BASED":
                frac = min(max((T - RULE_TEMP_LOW_C) / (RULE_TEMP_HIGH_C - RULE_TEMP_LOW_C), 0), 1)
                duty = 40 + frac * 60
            elif self.mode == "FIXED_PWM":
                duty = 50.2
            else:
                duty = 40 + max(0, (T + 8 * (T - prev) - 42)) * 2.2
            duty = min(duty, 100)
            heat = 0.020 * load
            cool = (0.010 + 0.030 * duty / 100) * (T - Tin)
            prev, T = T, T + heat - cool + random.gauss(0, 0.05)
            dTdt = T - prev
            fut = T + 10 * dTdt + random.gauss(0, 0.3)
            risk = min(max((T - 40) / (THERMAL_LIMIT_C - 40), 0), 1)
            p_mw = 1200 * (duty / 100) ** 1.6
            line = (f"DATA,{int(t*1000)},{self.mode},{T:.3f},{Tin + random.gauss(0, .05):.3f},"
                    f"{dTdt:.4f},{load:.2f},{duty:.2f},{duty + random.gauss(0, 4):.2f},"
                    f"{fut:.3f},{risk:.4f},1,{12*duty/100:.3f},{p_mw/12:.2f},{p_mw:.2f},1")
            self._handle(line)
            time.sleep(1.0)


def get_source(kind: str, port: str = "", baud: int = 115200, mode: str = "RULE_BASED"):
    """Keep one live source per browser session; replace it if settings change."""
    key = (kind, port, baud, mode)
    cur = st.session_state.get("src")
    if cur is not None and st.session_state.get("src_key") == key:
        return cur
    if cur is not None:
        cur.close()
    src = SerialSource(port, baud) if kind == "serial" else DemoSource(mode)
    st.session_state.src, st.session_state.src_key = src, key
    return src


def stop_source():
    cur = st.session_state.pop("src", None)
    st.session_state.pop("src_key", None)
    if cur is not None:
        cur.close()


# ---------------------------------------------------------------------------
# Charts (one y-axis per chart; thin lines; legend + hover on every chart)
# ---------------------------------------------------------------------------
def base_layout(fig: go.Figure, title: str, ytitle: str, height: int = 300):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)),
        height=height, margin=dict(l=10, r=10, t=48, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        xaxis=dict(title="time (s)", showgrid=False, zeroline=False),
        yaxis=dict(title=ytitle, gridcolor="rgba(128,128,128,0.18)", zeroline=False),
    )
    return fig


def line(fig, x, y, name, color, dash=None, shape="linear"):
    fig.add_trace(go.Scatter(
        x=x, y=y, name=name, mode="lines",
        line=dict(color=color, width=2, dash=dash, shape=shape),
        hovertemplate="%{y:.2f}",
    ))


def temp_chart(df: pd.DataFrame, show_limit: bool) -> go.Figure:
    fig = go.Figure()
    line(fig, df.t_s, df.proc_temp_c, "Processor temp", C_BLUE)
    line(fig, df.t_s, df.inlet_temp_c, "Inlet / ambient", C_ORANGE)
    if df.ann_future_temp_c.notna().any():
        line(fig, df.t_s, df.ann_future_temp_c, "ANN predicted (future)", C_AQUA, dash="dot")
    if show_limit:
        fig.add_hline(y=THERMAL_LIMIT_C, line=dict(color=C_MUTED, dash="dash", width=1),
                      annotation_text=f"Thermal limit {THERMAL_LIMIT_C:.0f} °C",
                      annotation_position="top left")
    return base_layout(fig, "Temperature", "°C")


def pwm_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    line(fig, df.t_s, df.applied_pwm_pct, "Applied PWM", C_BLUE, shape="hv")
    if df.ann_pwm_pct.notna().any():
        line(fig, df.t_s, df.ann_pwm_pct, "ANN suggested PWM", C_AQUA, dash="dot", shape="hv")
    fig = base_layout(fig, "Fan PWM duty", "% duty")
    fig.update_yaxes(range=[0, 105])
    return fig


def risk_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    line(fig, df.t_s, df.ann_thermal_risk, "Thermal risk", C_BLUE)
    fig = base_layout(fig, "ANN thermal-risk score", "risk (0–1)", height=240)
    fig.update_yaxes(range=[0, 1.02])
    fig.update_layout(showlegend=False)
    return fig


def dtdt_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    line(fig, df.t_s, df.dTdt_c_per_s, "dT/dt", C_BLUE)
    fig.add_hline(y=0, line=dict(color=C_MUTED, width=1))
    fig = base_layout(fig, "Rate of temperature change", "°C / s", height=240)
    fig.update_layout(showlegend=False)
    return fig


def fmt(v, unit="", nd=1):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}{unit}"


def kpi_row(df: pd.DataFrame):
    last = df.iloc[-1]
    c = st.columns(6)
    c[0].metric("Processor temp", fmt(last.proc_temp_c, " °C"),
                fmt(last.dTdt_c_per_s * 60, " °C/min") if not math.isnan(last.dTdt_c_per_s) else None,
                delta_color="inverse")
    c[1].metric("Inlet temp", fmt(last.inlet_temp_c, " °C"))
    c[2].metric("Fan PWM", fmt(last.applied_pwm_pct, " %", 0))
    c[3].metric("ANN predicted temp", fmt(last.ann_future_temp_c, " °C"))
    c[4].metric("Thermal risk", fmt(last.ann_thermal_risk, "", 2))
    c[5].metric("Controller", {"FIXED_PWM": "Fixed", "RULE_BASED": "Rule", "ANN": "ANN"}.get(str(last.control_mode), str(last.control_mode)))
    flags = []
    if last.sensor_valid < 0.5:
        flags.append("⚠️ **Sensor read invalid** — DS18B20 addresses in config.h are probably still placeholders.")
    if last.ann_valid < 0.5:
        flags.append("ℹ️ **ANN not running on the board** (TFLite Micro library not installed) — ANN lines are hidden.")
    if not math.isnan(last.proc_temp_c) and last.proc_temp_c >= THERMAL_LIMIT_C:
        flags.append(f"🛑 **Above thermal limit** ({THERMAL_LIMIT_C:.0f} °C) — firmware forces PWM to 100 %.")
    for f in flags:
        st.markdown(f)


def render_monitor(df: pd.DataFrame, window_s: int):
    if df.empty:
        st.info("Waiting for `DATA,` rows… (the fan self-test at boot takes ~25 s before the first row)")
        return
    view = df[df.t_s >= df.t_s.iloc[-1] - window_s] if window_s else df
    kpi_row(view)
    show_limit = view.proc_temp_c.max(skipna=True) > THERMAL_LIMIT_C - 15
    l, r = st.columns(2)
    l.plotly_chart(temp_chart(view, show_limit), use_container_width=True)
    r.plotly_chart(pwm_chart(view), use_container_width=True)
    l2, r2 = st.columns(2)
    if view.ann_thermal_risk.notna().any():
        l2.plotly_chart(risk_chart(view), use_container_width=True)
    r2.plotly_chart(dtdt_chart(view), use_container_width=True)
    with st.expander("Data table (last 200 rows)"):
        st.dataframe(view.tail(200)[COLUMNS], use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
DEFAULT_POWER_TABLE = pd.DataFrame({"pwm_pct": [0, 40, 70, 100], "power_w": [0.0, 0.35, 0.75, 1.20]})


def fan_power_w(df: pd.DataFrame, table: pd.DataFrame) -> pd.Series:
    """Measured INA219 power if the run has it, else interpolate the user's
    multimeter table (PWM % -> watts)."""
    if "fan_power_mw" in df and (df.fan_power_mw > 0).any():
        return df.fan_power_mw / 1000.0
    t = table.dropna().sort_values("pwm_pct")
    return pd.Series(np.interp(df.applied_pwm_pct, t.pwm_pct, t.power_w), index=df.index)


def summarise(name: str, df: pd.DataFrame, table: pd.DataFrame) -> dict:
    dt = df.t_s.diff().fillna(0).clip(upper=5)
    p = fan_power_w(df, table)
    dur = df.t_s.iloc[-1] - df.t_s.iloc[0]
    return {
        "run": name,
        "mode": df.control_mode.mode().iat[0] if df.control_mode.notna().any() else "?",
        "duration_min": dur / 60,
        "mean_temp_c": df.proc_temp_c.mean(),
        "peak_temp_c": df.proc_temp_c.max(),
        "temp_std_c": df.proc_temp_c.std(),
        "mean_pwm_pct": df.applied_pwm_pct.mean(),
        "mean_fan_power_w": p.mean(),
        "fan_energy_wh": float((p * dt).sum() / 3600.0),
        "pct_time_above_rule_high": float((df.proc_temp_c >= RULE_TEMP_HIGH_C).mean() * 100),
        "power_source": "INA219" if (df.get("fan_power_mw", pd.Series([0])) > 0).any() else "table estimate",
    }


def bar(summary: pd.DataFrame, col: str, title: str, unit: str) -> go.Figure:
    colors = [MODE_COLORS.get(m, C_MUTED) for m in summary["mode"]]
    fig = go.Figure(go.Bar(
        x=summary["run"], y=summary[col], marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.2f}" for v in summary[col]], textposition="outside",
        hovertemplate="%{x}<br>%{y:.3f} " + unit + "<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=14)), height=280,
        margin=dict(l=10, r=10, t=44, b=10), bargap=0.45, showlegend=False,
        yaxis=dict(title=unit, gridcolor="rgba(128,128,128,0.18)", rangemode="tozero"),
        xaxis=dict(showgrid=False),
    )
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🌀 ANN-Based Adaptive PWM Cooling — Dashboard")

tab_live, tab_cmp, tab_model = st.tabs(["📡 Live monitor", "⚖️ Controller comparison", "🧠 Model performance"])

# ---------------- Live monitor ----------------
with tab_live:
    with st.sidebar:
        st.header("Live data source")
        options = ["Demo (simulated)", "ESP32 serial port", "Open a saved log"]
        if not HAVE_SERIAL:
            options.remove("ESP32 serial port")
            st.caption("Install `pyserial` to read the board directly.")
        source = st.radio("Source", options, index=0)
        window_label = st.selectbox("Time window", ["Last 2 min", "Last 5 min", "Last 15 min", "Whole run"], index=1)
        window_s = {"Last 2 min": 120, "Last 5 min": 300, "Last 15 min": 900, "Whole run": 0}[window_label]

        port, baud, demo_mode = "", 115200, "RULE_BASED"
        if source == "ESP32 serial port":
            ports = [p.device for p in serial.tools.list_ports.comports()]
            if ports:
                port = st.selectbox("Port", ports, help="Close the PlatformIO Serial Monitor first — only one program can hold the port.")
            else:
                port = st.text_input("Port", "COM3", help="No ports detected. Type it (e.g. COM3 or /dev/ttyUSB0).")
            baud = st.number_input("Baud", value=115200, step=9600)
        elif source == "Demo (simulated)":
            demo_mode = st.selectbox("Simulated controller", ["RULE_BASED", "FIXED_PWM", "ANN"])

        running = st.toggle("Running", value=True, disabled=source == "Open a saved log")

    if source == "Open a saved log":
        stop_source()
        up = st.file_uploader("Serial Monitor capture or logs/run_*.csv", type=["csv", "txt", "log"], key="live_up")
        if up:
            render_monitor(parse_log_text(up.getvalue().decode("utf-8", errors="replace")), window_s)
    elif not running:
        stop_source()
        st.info("Paused. Turn **Running** back on in the sidebar.")
    else:
        kind = "serial" if source == "ESP32 serial port" else "demo"
        src = get_source(kind, port, int(baud), demo_mode)

        @st.fragment(run_every=1.0)
        def live_panel():
            df, msgs = src.snapshot()
            if src.error:
                st.error(f"Serial error on {port}: {src.error}")
            render_monitor(df, window_s)
            c1, c2 = st.columns([3, 1])
            with c1.expander("Board messages (# lines)"):
                st.code("\n".join(msgs[-20:]) or "—", language=None)
            c2.caption(f"Saving to `{src.log_path.relative_to(HERE)}` · {len(df)} rows")
            if not df.empty:
                c2.download_button("Download this run (CSV)", df[COLUMNS].to_csv(index=False),
                                   file_name=src.log_path.name, mime="text/csv",
                                   key=f"dl_{len(df)}")

        live_panel()

# ---------------- Comparison ----------------
with tab_cmp:
    st.markdown("Load one log per controller run (FIXED_PWM, RULE_BASED, ANN) — same workload level, "
                "same starting temperature — to compare them side by side.")
    files = st.file_uploader("Run logs", type=["csv", "txt", "log"], accept_multiple_files=True, key="cmp_up")
    with st.expander("Fan power vs PWM (used when a run has no INA219 readings)", expanded=False):
        st.caption("Replace these placeholder numbers with your multimeter readings: "
                   "fan supply volts × amps at each fixed PWM level.")
        power_table = st.data_editor(DEFAULT_POWER_TABLE, num_rows="dynamic", hide_index=True,
                                     column_config={"pwm_pct": "PWM %", "power_w": "Fan power (W)"})

    runs = {}
    for f in files or []:
        d = parse_log_text(f.getvalue().decode("utf-8", errors="replace"))
        if d.empty:
            st.warning(f"{f.name}: no DATA rows found.")
        else:
            runs[Path(f.name).stem] = d

    if not runs:
        st.info("No runs loaded yet. Logs saved by the Live monitor are in `dashboard/logs/`.")
    else:
        summary = pd.DataFrame([summarise(n, d, power_table) for n, d in runs.items()])
        order = {m: i for i, m in enumerate(MODE_COLORS)}
        summary = summary.sort_values("mode", key=lambda s: s.map(order).fillna(9)).reset_index(drop=True)
        runs = {n: runs[n] for n in summary["run"]}
        k = st.columns(4)
        k[0].plotly_chart(bar(summary, "mean_temp_c", "Mean processor temp", "°C"), use_container_width=True)
        k[1].plotly_chart(bar(summary, "peak_temp_c", "Peak processor temp", "°C"), use_container_width=True)
        k[2].plotly_chart(bar(summary, "mean_pwm_pct", "Mean fan PWM", "%"), use_container_width=True)
        k[3].plotly_chart(bar(summary, "fan_energy_wh", "Fan energy used", "Wh"), use_container_width=True)
        st.caption("Bar colour = controller: " + " · ".join(
            f"<span style='color:{c}'>■</span> {m}" for m, c in MODE_COLORS.items()), unsafe_allow_html=True)

        st.dataframe(summary.round(3), use_container_width=True, hide_index=True)
        if (summary.power_source == "table estimate").any():
            st.caption("Energy for runs without INA219 data is estimated from the PWM→power table above, "
                       "not measured per sample — say so in the write-up.")

        # Overlaid traces, aligned to each run's start
        fig_t, fig_p = go.Figure(), go.Figure()
        for (name, d), mode in zip(runs.items(), summary["mode"]):
            col = MODE_COLORS.get(mode, C_MUTED)
            line(fig_t, d.t_s, d.proc_temp_c, name, col)
            line(fig_p, d.t_s, d.applied_pwm_pct, name, col, shape="hv")
        a, b = st.columns(2)
        a.plotly_chart(base_layout(fig_t, "Processor temperature by run", "°C"), use_container_width=True)
        b.plotly_chart(base_layout(fig_p, "Fan PWM by run", "% duty"), use_container_width=True)
        st.download_button("Download comparison table (CSV)", summary.to_csv(index=False),
                           file_name="controller_comparison.csv", mime="text/csv")

# ---------------- Model performance ----------------
with tab_model:
    st.markdown("**INT8 model, test-set error** (full 122-file BMCDATA dataset, from the project README)")
    st.dataframe(pd.DataFrame({
        "Output": ["future_temp_c", "thermal_risk_score (0–1)", "pwm_pct"],
        "Test MAE": ["0.57 °C", "0.05", "12.4 % points"],
        "Trust it?": ["✅ Yes", "✅ Reasonable", "❌ No — only 4 PWM levels in the dataset"],
    }), hide_index=True, use_container_width=True)

    captions = {
        "pred_vs_actual_future_temp_c.png": "Predicted vs actual — future temperature",
        "pred_vs_actual_thermal_risk_score.png": "Predicted vs actual — thermal risk",
        "pred_vs_actual_pwm_pct.png": "Predicted vs actual — PWM %",
        "workload_vs_temp.png": "Workload vs temperature (training data)",
        "dTdt_vs_pwm.png": "dT/dt vs PWM (training data)",
        "fan_mode_breakdown.png": "Fan-mode breakdown of the dataset",
    }
    imgs = [p for p in (PLOTS_DIR / n for n in captions) if p.exists()]
    imgs += [p for p in sorted(PLOTS_DIR.glob("*.png")) if p.name not in captions]
    if not imgs:
        st.info(f"No plots found in `{PLOTS_DIR}`. Run `python/visualize.py` to generate them.")
    for i in range(0, len(imgs), 2):
        cols = st.columns(2)
        for c, p in zip(cols, imgs[i:i + 2]):
            c.image(str(p), caption=captions.get(p.name, p.stem), use_container_width=True)

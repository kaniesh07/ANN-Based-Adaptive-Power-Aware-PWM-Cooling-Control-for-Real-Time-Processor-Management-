"""
bmcdata_loader.py

Loads the real BMCDATA dataset (https://github.com/arealuser/bmcdata,
DOI 10.21227/6yep-by05 on IEEE DataPort) from its native InfluxDB-exported
CSV format.

Each raw file in data/ is NOT a normal CSV. It's two pivoted InfluxDB
exports concatenated together -- one `_measurement=bmc` section (board
sensors: CPU/inlet/outlet temps, fan RPM, PSU power) and one
`_measurement=host` section (OS-level perf counters, cpu_usage, per-core
utilization, etc), each with its own `#group`/`#datatype`/`#default`
metadata header. This loader:

  1. Splits a raw file into its bmc/host sections (same idea as the
     dataset's own scripts/split.py, done more robustly here: by actually
     locating `#group` lines rather than assuming a fixed line number).
  2. Coerces numeric columns, parses `_time`.
  3. merge_asof's bmc + host on `_time` (nearest), matching
     scripts/collapse.py's approach.
  4. Attaches per-file experiment metadata (fan mode, workload) from
     bmcdata_inventory.csv, which is a straight copy of the "Dataset
     Inventory" table in the dataset's own README -- that metadata is
     NOT present in the CSV rows themselves, it's file-level.

Usage:
    from bmcdata_loader import load_dataset
    df = load_dataset(data_dir="bmcdata_repo/data", inventory_csv="bmcdata_inventory.csv")
"""

import os
import re

import pandas as pd

# Columns InfluxDB's export always includes that carry no data-content value.
META_COLS_TO_DROP = {"result", "table", "_start", "_stop", "_measurement", "device_id"}


def _read_sections(path: str):
    """Split a raw BMCDATA file into its pivoted sections (one per
    `#group` block), returning {measurement_name: DataFrame}."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    group_starts = [i for i, l in enumerate(lines) if l.startswith("#group")]
    if not group_starts:
        raise ValueError(f"{path}: no '#group' lines found -- not a BMCDATA export?")
    group_starts.append(len(lines))  # sentinel end

    sections = {}
    for gi in range(len(group_starts) - 1):
        start, end = group_starts[gi], group_starts[gi + 1]
        block = lines[start:end]
        # header is 3 lines after '#group' (#group, #datatype, #default, HEADER, data...)
        header_line = block[3].rstrip("\r\n")
        data_lines = [l.rstrip("\r\n") for l in block[4:] if l.strip()]
        if not data_lines:
            continue

        header = header_line.split(",")
        rows = [l.split(",") for l in data_lines]
        # Guard against the rare trailing-blank-line / malformed-row case.
        rows = [r for r in rows if len(r) == len(header)]
        df = pd.DataFrame(rows, columns=header)

        # NOTE: `_measurement` is a constant project codename in this dataset
        # (always "sdgp") -- it's `device_id` that actually distinguishes the
        # "bmc" section from the "host" section. Keying off `_measurement`
        # (as you might reasonably expect from the column name) silently
        # merges both sections into one "sdgp" bucket -- keep this keyed on
        # device_id.
        key = df["device_id"].iloc[0] if "device_id" in df.columns else f"section{gi}"
        sections[key] = df

    return sections


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_time"] = pd.to_datetime(df["_time"], errors="coerce", utc=True)
    keep_cols = [c for c in df.columns if c not in META_COLS_TO_DROP and c != "" and not c.startswith("Unnamed")]
    df = df[keep_cols]
    for c in df.columns:
        if c == "_time":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Fan-mode / workload metadata parsing (file-level, from the inventory CSV)
# ---------------------------------------------------------------------------

def parse_fan_mode(raw: str):
    """Returns (fan_mode_type, fixed_pwm_pct_or_None, pid_setpoint_c_or_None).

    fan_mode_type is one of:
      FIXED_PWM      -- single fixed PWM value for the whole file (clean label)
      FIXED_PWM_RANGE-- PWM swept within the file (e.g. "30/255~255/255") --
                        NOT a clean per-row label since we don't know which
                        value applied at which timestamp from this column
                        alone; excluded from direct PWM supervision.
      PID            -- PID control targeting an inlet temperature setpoint
      TEAP / HYPERFAN / OTHER -- learned/heuristic controllers, no directly
                        observable PWM value in this dataset's columns
    """
    s = raw.strip()

    if "~" in s and "PWM" in s:
        return "FIXED_PWM_RANGE", None, None

    m = re.match(r"^PWM\s+(\d+)\s*/\s*255$", s)
    if m:
        pwm_byte = int(m.group(1))
        return "FIXED_PWM", pwm_byte / 255.0 * 100.0, None

    m = re.match(r"^PID\s+([\d.]+)", s)
    if m:
        return "PID", None, float(m.group(1))

    if s.upper().startswith("TEAP"):
        return "TEAP", None, None
    if s.upper().startswith("HYPERFAN"):
        return "HYPERFAN", None, None
    if s.upper() == "NONE":
        return "NONE", 0.0, None

    return "OTHER", None, None


def load_inventory(inventory_csv: str) -> pd.DataFrame:
    inv = pd.read_csv(inventory_csv)
    parsed = inv["fan_mode_raw"].apply(parse_fan_mode)
    inv["fan_mode_type"] = [p[0] for p in parsed]
    inv["fixed_pwm_pct"] = [p[1] for p in parsed]
    inv["pid_setpoint_c"] = [p[2] for p in parsed]
    return inv


# ---------------------------------------------------------------------------
# Per-file loading + merge
# ---------------------------------------------------------------------------

def load_file(path: str, meta_row: pd.Series) -> pd.DataFrame:
    sections = _read_sections(path)
    if "bmc" not in sections or "host" not in sections:
        raise ValueError(f"{path}: expected 'bmc' and 'host' sections, found {list(sections.keys())}")

    bmc = _coerce_numeric(sections["bmc"]).sort_values("_time")
    host = _coerce_numeric(sections["host"]).sort_values("_time")

    # A malformed/truncated row occasionally leaves `_time` as NaT after
    # coercion (seen in at least one real file in this dataset) -- merge_asof
    # requires a fully non-null join key, so drop those rows first rather
    # than failing the whole file.
    bmc = bmc.dropna(subset=["_time"])
    host = host.dropna(subset=["_time"])

    merged = pd.merge_asof(host, bmc, on="_time", direction="nearest", suffixes=("", "_bmc"))

    extra = pd.DataFrame({
        "source_file": os.path.basename(path),
        "fan_mode_raw": meta_row["fan_mode_raw"],
        "fan_mode_type": meta_row["fan_mode_type"],
        "fixed_pwm_pct": meta_row["fixed_pwm_pct"],
        "pid_setpoint_c": meta_row["pid_setpoint_c"],
        "workload_raw": meta_row["workload_raw"],
        "method": meta_row["method"],
    }, index=merged.index)
    merged = pd.concat([merged, extra], axis=1)
    return merged


def load_dataset(data_dir: str, inventory_csv: str, files: list = None) -> pd.DataFrame:
    """Loads every file listed in the inventory (or just `files`, a subset
    of filenames, if given) and concatenates them into one long DataFrame.
    Missing files are skipped with a warning -- the full dataset is ~35MB
    across 124 files; a subset is fine for a first pipeline run."""
    inv = load_inventory(inventory_csv)
    if files is not None:
        inv = inv[inv["file"].isin(files)]

    frames = []
    for _, row in inv.iterrows():
        path = os.path.join(data_dir, row["file"])
        if not os.path.exists(path):
            print(f"[bmcdata_loader] WARNING: {path} not found, skipping")
            continue
        try:
            frames.append(load_file(path, row))
        except Exception as e:
            print(f"[bmcdata_loader] WARNING: failed to load {path}: {e}")

    if not frames:
        raise FileNotFoundError(
            f"No BMCDATA files loaded from {data_dir}. "
            "Clone https://github.com/arealuser/bmcdata and point --data-dir at its data/ folder."
        )

    df = pd.concat(frames, ignore_index=True)
    print(f"[bmcdata_loader] loaded {len(frames)} file(s), {len(df)} merged rows")
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../bmcdata_repo/data")
    ap.add_argument("--inventory", default="bmcdata_inventory.csv")
    ap.add_argument("--limit", type=int, default=5, help="load only the first N files, for a quick smoke test")
    args = ap.parse_args()

    inv = load_inventory(args.inventory)
    sample_files = inv["file"].head(args.limit).tolist()
    df = load_dataset(args.data_dir, args.inventory, files=sample_files)
    print(df[["_time", "Cpu1_Temp", "Cpu2_Temp", "Inlet_Temp", "cpu_usage",
              "fan_mode_type", "fixed_pwm_pct", "source_file"]].head(10))
    print(df["fan_mode_type"].value_counts())

from __future__ import annotations

import os
import re
import signal
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Granular_MPH GUI v5", layout="wide", initial_sidebar_state="expanded")

# =============================================================================
# CSS
# =============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

* { font-family: 'Inter', sans-serif !important; }
code, .stTextArea textarea, .stCode { font-family: 'IBM Plex Mono', monospace !important; }

.main .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1600px; }

/* ===== HERO ===== */
.hero {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 1.4rem 1.6rem 1.1rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(56,189,248,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero h1 { margin: 0 0 0.3rem; font-size: 1.8rem; font-weight: 700; color: #f0f6fc; letter-spacing: -0.5px; }
.hero .sub { color: #8b949e; font-size: 0.9rem; line-height: 1.6; margin: 0; }
.hero .badge { display: inline-block; background: #1f6feb22; border: 1px solid #1f6feb55; color: #58a6ff; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; margin-right: 6px; }

/* ===== STEP PROGRESS BAR ===== */
.step-bar {
    display: flex; gap: 4px; margin: 0 0 1.2rem;
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 14px; padding: 0.6rem 0.8rem;
}
.step-item {
    flex: 1; display: flex; align-items: center; gap: 7px;
    padding: 6px 10px; border-radius: 10px;
    background: #161b22; border: 1px solid #21262d;
    cursor: default; transition: all 0.2s;
}
.step-item.active { background: #1c2d3f; border-color: #388bfd; }
.step-num {
    width: 22px; height: 22px; border-radius: 50%; display: flex;
    align-items: center; justify-content: center; font-size: 0.72rem;
    font-weight: 700; flex-shrink: 0;
    background: #21262d; color: #8b949e;
}
.step-item.active .step-num { background: #388bfd; color: #fff; }
.step-label { font-size: 0.72rem; color: #8b949e; line-height: 1.3; }
.step-item.active .step-label { color: #e6edf3; }
.step-label .en { display: block; font-weight: 600; }
.step-label .ja { display: block; color: #6e7681; font-size: 0.66rem; }
.step-item.active .step-label .ja { color: #8b949e; }

/* ===== SECTION HEADER ===== */
.sec-header {
    display: flex; align-items: center; gap: 10px;
    padding: 0.75rem 1rem 0.6rem;
    border-radius: 12px 12px 0 0;
    margin-bottom: 0;
}
.sec-num {
    width: 28px; height: 28px; border-radius: 50%; display: flex;
    align-items: center; justify-content: center;
    font-size: 0.85rem; font-weight: 700; color: #fff; flex-shrink: 0;
}
.sec-title { font-size: 1.05rem; font-weight: 700; color: #f0f6fc; }
.sec-subtitle { font-size: 0.8rem; color: #8b949e; margin-top: 1px; }

/* Step colors */
.c1 { background: #1a2e22; border: 1px solid #2ea04333; }
.c1 .sec-num { background: #2ea043; }
.c2 { background: #1e1f2e; border: 1px solid #7c3aed44; }
.c2 .sec-num { background: #7c3aed; }
.c3 { background: #1a2536; border: 1px solid #0284c744; }
.c3 .sec-num { background: #0284c7; }
.c4 { background: #2a1a1a; border: 1px solid #dc262644; }
.c4 .sec-num { background: #dc2626; }
.c5 { background: #1a2a29; border: 1px solid #0d9488aa; }
.c5 .sec-num { background: #0d9488; }
.c6 { background: #1e1e2e; border: 1px solid #6366f144; }
.c6 .sec-num { background: #6366f1; }

/* ===== CARDS ===== */
.stat-grid { display: flex; flex-wrap: wrap; gap: 8px; margin: 0.8rem 0; }
.stat-card {
    padding: 8px 14px; border-radius: 10px;
    border: 1px solid #21262d; background: #161b22;
    min-width: 180px;
}
.stat-card .slabel { font-size: 0.7rem; color: #6e7681; margin-bottom: 2px; }
.stat-card .sval { font-size: 0.82rem; color: #e6edf3; font-weight: 500; word-break: break-all; }
.stat-card.ok { border-color: #2ea04366; background: #0d1a12; }
.stat-card.ok .sval { color: #3fb950; }
.stat-card.ng { border-color: #f8511566; background: #1a1208; }
.stat-card.ng .sval { color: #f85149; }

/* ===== NOTICE BOX ===== */
.notice {
    padding: 10px 14px; border-radius: 10px; margin: 8px 0;
    font-size: 0.85rem; line-height: 1.6;
}
.notice.blue { background: #1c2d3f; border-left: 4px solid #388bfd; color: #cae8ff; }
.notice.green { background: #0d1a12; border-left: 4px solid #2ea043; color: #7ee787; }
.notice.yellow { background: #1f1a10; border-left: 4px solid #d29922; color: #e3b341; }

/* ===== BUTTON GROUPS ===== */
div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    border: none !important; color: #fff !important;
    font-weight: 600 !important; border-radius: 8px !important;
}
div[data-testid="stButton"] button:not([kind="primary"]) {
    background: #21262d !important; border: 1px solid #30363d !important;
    color: #e6edf3 !important; border-radius: 8px !important;
}

/* ===== DATA EDITOR FIX ===== */
div[data-testid="stDataEditor"] { border: 1px solid #30363d !important; border-radius: 10px; }

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1117; border-radius: 12px; gap: 2px; padding: 4px;
    border: 1px solid #21262d;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: #8b949e !important;
    font-size: 0.82rem !important; font-weight: 600 !important;
    padding: 6px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important; color: #f0f6fc !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1rem !important; }

/* ===== INPUTS ===== */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: #0d1117 !important; color: #e6edf3 !important;
    border: 1px solid #30363d !important; border-radius: 8px !important;
}
.stSelectbox > div > div { background: #0d1117 !important; border-color: #30363d !important; }

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d !important;
}
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .block-container { color: #e6edf3 !important; }

.help-text { font-size: 0.78rem; color: #6e7681; margin: 2px 0 8px; line-height: 1.5; }
.divider { border: none; border-top: 1px solid #21262d; margin: 0.8rem 0; }

/* ===== BILINGUAL / PROCESS / PREVIEW ===== */
.lang-pill {
    display:inline-block; padding:2px 8px; border-radius:999px;
    border:1px solid #30363d; background:#161b22; color:#8b949e;
    font-size:0.72rem; margin-left:6px;
}
.kill-box {
    padding:12px 14px; border-radius:12px;
    background:#2a1a1a; border:1px solid #dc262655; color:#ffd7d7;
    margin:8px 0;
}
.param-help {
    padding:8px 10px; border-radius:8px;
    background:#0d1117; border:1px solid #21262d;
    font-size:0.82rem; line-height:1.55; color:#c9d1d9;
}

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# UI helpers
# =============================================================================

def sec(num: int, en: str, ja: str = "", cls: str = "c1", sub_en: str = "", sub_ja: str = "") -> None:
    """English-only section header.

    The old GUI accepted both English and Japanese strings.  To keep the rest
    of the code compatible, this function still accepts the Japanese arguments
    but only displays the English text.
    """
    sub = f'<div class="sec-subtitle">{sub_en}</div>' if sub_en else ""
    st.markdown(
        f'<div class="sec-header {cls}">'
        f'<div class="sec-num">{num}</div>'
        f'<div><div class="sec-title">{en}</div>{sub}</div>'
        f'</div>', unsafe_allow_html=True)

def notice(msg: str, kind: str = "blue") -> None:
    st.markdown(f'<div class="notice {kind}">{msg}</div>', unsafe_allow_html=True)


# =============================================================================
# Bilingual labels / process control / preview / data help
# =============================================================================
def bi(en: str, ja: str = "") -> str:
    """Return the English label only."""
    return en

def help_bi(en: str, ja: str = "") -> str:
    return en

PARTICLE_TYPE_HELP = {
    0: ("Fluid 0 / primary liquid", "Fluid 0"),
    1: ("Fluid 1 / secondary liquid or gas", "Fluid 1"),
    2: ("Structure 1 / elastoplastic solid", "Structure 1"),
    3: ("Structure 2 / elastic or secondary solid", "Structure 2"),
    4: ("Wall / fixed boundary", "Wall"),
    5: ("Moving rigid body", "Moving rigid body"),
}

DATA_PARAM_HELP = {
    "Dt": ("Global time step. Smaller is more stable but slower.", "Global time step. Smaller is more stable but slower."),
    "Elastic_Dt": ("Sub time step for elastic/plastic calculation.", "Sub time step for elastic/plastic calculation."),
    "ElasticDt": ("Sub time step for elastic/plastic calculation. This is the current d.data spelling.", "Sub time step for elastic/plastic calculation."),
    "FinishTime": ("Physical end time of the simulation.", "Physical end time of the simulation."),
    "EndTime": ("Physical end time of the simulation. This is the current d.data spelling.", "Physical end time of the simulation."),
    "OutputInterval": ("Interval for writing profile/output files.", "Interval for writing profile/output files."),
    "VtkOutputInterval": ("Interval for writing VTK visualization files.", "Interval for writing VTK visualization files."),
    "ParticleDistance": ("Initial particle spacing.", "Initial particle spacing."),
    "Density": ("Material density.", "Material density."),
    "Viscosity": ("Dynamic or numerical viscosity depending on solver.", "Dynamic or numerical viscosity depending on solver."),
    "SurfaceTension": ("Surface tension row in strict d.data format; four values are preserved as written.", "Surface tension."),
    "ActualDebrisSize": ("Actual debris size row in strict d.data format; two values are preserved as written.", "Actual debris size."),
    "YoungModulus": ("Young's modulus for elastic/plastic material.", "Young's modulus for elastic/plastic material."),
    "PoissonRatio": ("Poisson's ratio.", "Poisson's ratio."),
    "InternalFrictionAngle": ("Friction angle for Drucker-Prager/Mohr-Coulomb plasticity.", "Friction angle for Drucker-Prager/Mohr-Coulomb plasticity."),
    "DilatancyFrictionAngle": ("Dilatancy angle controlling volumetric plastic flow.", "Dilatancy angle controlling volumetric plastic flow."),
    "Cohesion": ("Cohesion/yield strength parameter.", "Cohesion/yield strength parameter."),
    "Type": ("Particle/material type ID. See particle type table.", "Particle/material type ID. See particle type table."),
    "RigidType": ("Rigid body group ID. -1 means not rigid.", "Rigid body group ID. -1 means not rigid."),
    "Gravity": ("Gravity acceleration vector.", "Gravity acceleration vector."),
    "Wall": ("Wall/boundary setting.", "Wall/boundary setting."),
    "Wall2": ("Motion setting for wall/rigid type 2: center, velocity, and angular velocity.", "Wall motion setting."),
    "Wall3": ("Motion setting for wall/rigid type 3: center, velocity, and angular velocity.", "Wall motion setting."),
    "InteractionRatio": ("Pairwise interaction multiplier between particle types.", "Interaction ratio."),
}

def describe_data_line(line: str) -> Tuple[str, str, str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("//"):
        return "", "", ""
    key = re.split(r"\s+", stripped, maxsplit=1)[0]
    value = re.split(r"\s+", stripped, maxsplit=1)[1] if len(re.split(r"\s+", stripped, maxsplit=1)) > 1 else ""
    # Exact first, then fuzzy contains
    if key in DATA_PARAM_HELP:
        en, ja = DATA_PARAM_HELP[key]
        return key, value, en
    for k, (en, ja) in DATA_PARAM_HELP.items():
        if k.lower() in key.lower():
            return key, value, en
    return key, value, ""

def data_help_dataframe(raw: str) -> pd.DataFrame:
    rows = []
    for i, line in enumerate(raw.splitlines(), 1):
        key, value, desc = describe_data_line(line)
        if key:
            rows.append({"Line": i, "Parameter": key, "Value": value, "Description": desc})
    return pd.DataFrame(rows)

def add_help_comments_to_data(raw: str) -> str:
    out = []
    for line in raw.splitlines():
        key, _, desc = describe_data_line(line)
        if desc and not (out and out[-1].startswith("# GUI help:") and key in out[-1]):
            out.append(f"# GUI help: {key}: {desc}")
        out.append(line)
    return "\n".join(out) + ("\n" if raw.endswith("\n") else "")


# =============================================================================
# Strict d.data global-parameter helpers
# =============================================================================
# These rows are not material arrays. They must be edited separately and written
# back in the same simple key-value style used by the user's d.data file.
DDATA_GLOBAL_ROWS = [
    ("Dt", "Global time step"),
    ("ElasticDt", "Elastic/plastic sub time step"),
    ("OutputInterval", "Profile/output interval"),
    ("VtkOutputInterval", "VTK output interval"),
    ("EndTime", "Simulation end time"),
    ("RadiusRatioA", "Neighbor radius ratio for acceleration/force"),
    ("RadiusRatioP", "Neighbor radius ratio for pressure"),
    ("RadiusRatioV", "Neighbor radius ratio for viscosity"),
    ("Gravity", "Gravity vector, e.g. 0.0 -9.81 0.0"),
    ("Wall2", "Wall/rigid motion: Center ... Velocity ... Omega ..."),
    ("Wall3", "Wall/rigid motion: Center ... Velocity ... Omega ..."),
]

DDATA_GLOBAL_DEFAULTS = {
    "Dt": "1.0e-4",
    "ElasticDt": "1.0e-4",
    "OutputInterval": "1.0",
    "VtkOutputInterval": "1.0e-2",
    "EndTime": "2.0e1",
    "RadiusRatioA": "2.5",
    "RadiusRatioP": "2.5",
    "RadiusRatioV": "2.5",
    "Gravity": "0.0 -9.81 0.0",
    "Wall2": "Center 0.0 0.0 0.0    Velocity 0.0  0.0 0.0    Omega 0.0 0.0 0.0",
    "Wall3": "Center 0.0 0.0 0.0    Velocity 0.0  0.0 0.0    Omega 0.0 0.0 0.0",
}

def _line_matches_key(line: str, key: str) -> bool:
    clean = line.strip()
    if not clean or clean.startswith("#") or clean.startswith("//"):
        return False
    parts = re.split(r"\s+", clean, maxsplit=1)
    return bool(parts and parts[0] == key)

def extract_ddata_global_value(raw: str, key: str) -> str:
    """Return the value after a strict d.data key, preserving the user's raw text."""
    for line in raw.splitlines():
        if _line_matches_key(line, key):
            parts = re.split(r"\s+", line.strip(), maxsplit=1)
            return parts[1].strip() if len(parts) > 1 else ""
    return DDATA_GLOBAL_DEFAULTS.get(key, "")

def build_global_table_from_data(raw: str) -> pd.DataFrame:
    rows = []
    for key, desc in DDATA_GLOBAL_ROWS:
        rows.append({"Parameter": key, "Value": extract_ddata_global_value(raw, key), "Description": desc})
    return pd.DataFrame(rows)

def apply_global_table_to_data(raw: str, global_df: pd.DataFrame) -> str:
    """Apply Dt/ElasticDt/.../Gravity/Wall2/Wall3 without touching material rows.

    This is intentionally strict for the user's d.data format:
      Key<TAB><TAB><TAB>Value
    Existing rows are replaced in place. Missing rows are inserted after the
    initial ####### line if it exists; otherwise they are prepended.
    """
    values: Dict[str, str] = {}
    for _, r in global_df.iterrows():
        key = str(r.get("Parameter", "")).strip()
        if key not in dict(DDATA_GLOBAL_ROWS):
            continue
        val = str(r.get("Value", "")).strip()
        values[key] = val

    lines = raw.splitlines()
    found = set()
    out = []
    for line in lines:
        replaced = False
        for key, val in values.items():
            if _line_matches_key(line, key):
                if key.startswith("Wall"):
                    out.append(f"{key}    {val}")
                else:
                    out.append(f"{key}\t\t\t{val}")
                found.add(key)
                replaced = True
                break
        if not replaced:
            out.append(line)

    missing = [(key, values[key]) for key, _ in DDATA_GLOBAL_ROWS if key in values and key not in found]
    if missing:
        insert_at = 1 if out and out[0].strip().startswith("####") else 0
        missing_lines = []
        for key, val in missing:
            if key.startswith("Wall"):
                missing_lines.append(f"{key}    {val}")
            else:
                missing_lines.append(f"{key}\t\t\t{val}")
        out = out[:insert_at] + missing_lines + out[insert_at:]

    return "\n".join(out) + ("\n" if raw.endswith("\n") else "")


# =============================================================================
# .data material table helpers
# =============================================================================
MATERIAL_ROWS = [
    {"Type": 0, "Role": "Fluid 0", "Use": True},
    {"Type": 1, "Role": "Fluid 1", "Use": True},
    {"Type": 2, "Role": "Structure 1", "Use": True},
    {"Type": 3, "Role": "Structure 2", "Use": True},
    {"Type": 4, "Role": "Wall / fixed boundary", "Use": True},
    {"Type": 5, "Role": "Moving wall / rigid body", "Use": True},
]

MATERIAL_NUMERIC_COLUMNS = [
    # Strict d.data general/fluid rows: Type 0..5
    "Density",
    "BulkModulus",
    "BulkViscosity",
    "ShearViscosity",
    # Strict d.data special rows
    "SurfaceTension",       # existing file format: four values, kept as Type 0..3
    "ActualDebrisSize",     # existing file format: two values, kept as Type 2..3
    # Strict d.data solid/wall rows: Type 2..5
    "YoungModulus",
    "PoissonRatio",
    "Cohesion",
    "InternalFrictionAngle",
    "DilatancyFrictionAngle",
]

# Strict reference layout for your d.data format.
# The GUI must preserve this style exactly instead of guessing another format.
#
# General/fluid rows:
#   Density              Type0 Type1 Type2 Type3 Type4 Type5
#   BulkModulus          Type0 Type1 Type2 Type3 Type4 Type5
#   BulkViscosity        Type0 Type1 Type2 Type3 Type4 Type5
#   ShearViscosity       Type0 Type1 Type2 Type3 Type4 Type5
# Special rows:
#   SurfaceTension       Type0 Type1 Type2 Type3
#   ActualDebrisSize     Type2 Type3
# Solid/wall rows:
#   YoungModulus         Type2 Type3 Type4 Type5
#   PoissonRatio         Type2 Type3 Type4 Type5
#   Cohesion             Type2 Type3 Type4 Type5
#   InternalFrictionAngle Type2 Type3 Type4 Type5
#   DilatancyFrictionAngle Type2 Type3 Type4 Type5
D_DATA_ARRAY_LAYOUTS = {
    "Density": (0, 6),
    "BulkModulus": (0, 6),
    "BulkViscosity": (0, 6),
    "ShearViscosity": (0, 6),
    "SurfaceTension": (0, 4),
    "ActualDebrisSize": (2, 2),
    "YoungModulus": (2, 4),
    "PoissonRatio": (2, 4),
    "Cohesion": (2, 4),
    "InternalFrictionAngle": (2, 4),
    "DilatancyFrictionAngle": (2, 4),
}

# In your d.data files, fluid/general parameters have six values for
# Type 0..5, while solid/plastic parameters have only four values for Type 2..5:
#   YoungModulus            E2 E3 E4 E5
# This set tells the parser/writer to preserve that offset instead of shifting
# E2 into Type 0.
SOLID_ONLY_MATERIAL_COLUMNS = {
    "YoungModulus",
    "PoissonRatio",
    "Cohesion",
    "InternalFrictionAngle",
    "DilatancyFrictionAngle",
}

def infer_array_start_index(param: str, n_values: int, max_type: int = 5) -> int:
    """Infer whether an array-style .data row starts at Type 0 or Type 2.

    Examples:
      Density       v0 v1 v2 v3 v4 v5  -> start 0
      YoungModulus        E2 E3 E4 E5  -> start 2

    This avoids a dangerous GUI bug where solid-only arrays are shifted into
    fluid material columns.
    """
    if param in D_DATA_ARRAY_LAYOUTS:
        expected_start, expected_count = D_DATA_ARRAY_LAYOUTS[param]
        if n_values == expected_count:
            return expected_start
    if param in SOLID_ONLY_MATERIAL_COLUMNS and n_values == max_type - 1:
        return 2
    return 0

MATERIAL_DEFAULTS = {
    0: {"Density": 1000.0, "BulkModulus": 2.0e6, "BulkViscosity": 0.0, "ShearViscosity": 1.0e-3, "SurfaceTension": 0.072},
    1: {"Density": 1.2,    "BulkModulus": 1.0e5, "BulkViscosity": 0.0, "ShearViscosity": 1.8e-5, "SurfaceTension": 0.072},
    2: {"Density": 2500.0, "SurfaceTension": 0.0, "ActualDebrisSize": 0.001, "YoungModulus": 1.0e6, "PoissonRatio": 0.30, "Cohesion": 0.0,
        "InternalFrictionAngle": 40.0, "DilatancyFrictionAngle": 0.0},
    3: {"Density": 2500.0, "SurfaceTension": 0.0, "ActualDebrisSize": 0.001, "YoungModulus": 1.0e7, "PoissonRatio": 0.30, "Cohesion": 0.0,
        "InternalFrictionAngle": 40.0, "DilatancyFrictionAngle": 0.0},
    4: {"Density": 2500.0, "YoungModulus": 1.0e9, "PoissonRatio": 0.30, "Cohesion": 0.0,
        "InternalFrictionAngle": 0.0,  "DilatancyFrictionAngle": 0.0},
    5: {"Density": 2500.0, "YoungModulus": 1.0e9, "PoissonRatio": 0.30, "Cohesion": 0.0,
        "InternalFrictionAngle": 0.0,  "DilatancyFrictionAngle": 0.0},
}

def _strip_inline_comment(line: str) -> str:
    # Keep this conservative: many existing .data files use comments after values.
    for token in ["//", "#"]:
        if token in line:
            line = line.split(token, 1)[0]
    return line.strip()

def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def _format_number_for_data(x) -> str:
    if x is None or x == "":
        return ""
    try:
        return f"{float(x):.10g}"
    except Exception:
        return str(x)

def _line_key_and_index(first_token: str) -> Tuple[str, int | None]:
    # Accept Density[0], Density(0), or Density0 style.
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[|\()(\d+)(?:\]|\))$", first_token)
    if m:
        return m.group(1), int(m.group(2))
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*?)(\d+)$", first_token)
    if m:
        return m.group(1), int(m.group(2))
    return first_token, None

def extract_material_values(raw: str, param: str, max_type: int = 5) -> Dict[int, float]:
    """Read values from common .data styles.

    Supported styles:
      1) Density 1000 1.2 6100 1000 1000 6000       -> Type 0..5
      2) YoungModulus 1e7 1e9 1e6 1e6               -> Type 2..5
      3) Density 0 1000
      4) Density[0] 1000
      5) Density0 1000

    The key improvement for d.data is style (2): solid-only rows with four
    values are interpreted as Type 2, Type 3, Type 4, Type 5.
    """
    values: Dict[int, float] = {}
    param_lower = param.lower()
    for line in raw.splitlines():
        clean = _strip_inline_comment(line)
        if not clean:
            continue
        parts = re.split(r"\s+", clean)
        key0, idx0 = _line_key_and_index(parts[0])
        if key0.lower() != param_lower:
            continue

        if idx0 is not None and len(parts) >= 2 and _is_number(parts[1]):
            values[idx0] = float(parts[1])
            continue

        # Indexed style: Density 0 1000
        if len(parts) >= 3 and parts[1].isdigit() and _is_number(parts[2]):
            idx = int(parts[1])
            values[idx] = float(parts[2])
            continue

        # Array style: Density v0 v1 v2 ... OR YoungModulus E2 E3 E4 E5
        arr = parts[1:]
        if arr and all(_is_number(v) for v in arr):
            start_idx = infer_array_start_index(param, len(arr), max_type=max_type)
            for offset, val in enumerate(arr):
                idx = start_idx + offset
                if idx <= max_type:
                    values[idx] = float(val)
    return values

def build_material_table_from_data(raw: str) -> pd.DataFrame:
    rows = []
    values_by_param = {p: extract_material_values(raw, p, max_type=5) for p in MATERIAL_NUMERIC_COLUMNS}
    for base in MATERIAL_ROWS:
        typ = int(base["Type"])
        row = dict(base)
        for p in MATERIAL_NUMERIC_COLUMNS:
            if typ in values_by_param[p]:
                row[p] = values_by_param[p][typ]
            else:
                row[p] = MATERIAL_DEFAULTS.get(typ, {}).get(p, "")
        rows.append(row)
    return pd.DataFrame(rows)

def detect_data_param_style(raw: str, param: str) -> str:
    """Return 'array', 'indexed', 'bracket', 'suffix', or 'missing'."""
    param_lower = param.lower()
    for line in raw.splitlines():
        clean = _strip_inline_comment(line)
        if not clean:
            continue
        parts = re.split(r"\s+", clean)
        key0, idx0 = _line_key_and_index(parts[0])
        if key0.lower() != param_lower:
            continue
        if "[" in parts[0] or "(" in parts[0]:
            return "bracket"
        if idx0 is not None:
            return "suffix"
        if len(parts) >= 3 and parts[1].isdigit() and _is_number(parts[2]):
            return "indexed"
        if len(parts) >= 2 and all(_is_number(v) for v in parts[1:]):
            return "array"
    return "missing"

def detect_data_param_array_start(raw: str, param: str, max_type: int = 5) -> int:
    """Return array start index for a parameter in the existing file.

    For d.data, solid-only rows often contain only Type 2..5 values.
    The writer uses this to preserve the original compact format.
    """
    param_lower = param.lower()
    for line in raw.splitlines():
        clean = _strip_inline_comment(line)
        if not clean:
            continue
        parts = re.split(r"\s+", clean)
        key0, idx0 = _line_key_and_index(parts[0])
        if key0.lower() != param_lower:
            continue
        if len(parts) >= 2 and all(_is_number(v) for v in parts[1:]):
            return infer_array_start_index(param, len(parts) - 1, max_type=max_type)
    return 0

def _values_from_material_df(df: pd.DataFrame, param: str) -> Dict[int, str]:
    values: Dict[int, str] = {}
    for _, r in df.iterrows():
        try:
            typ = int(r["Type"])
        except Exception:
            continue
        if "Use" in r and bool(r["Use"]) is False:
            continue
        val = r.get(param, "")
        if val == "" or pd.isna(val):
            continue
        values[typ] = _format_number_for_data(val)
    return values

def apply_material_table_to_data(raw: str, material_df: pd.DataFrame) -> str:
    """Apply the material table to the raw .data text.

    Existing line style is preserved for each parameter if possible.  Missing
    parameters are appended under a GUI-managed block in indexed form, e.g.
    `Density 0 1000`.
    """
    lines = raw.splitlines()
    append_lines: List[str] = []
    for param in MATERIAL_NUMERIC_COLUMNS:
        values = _values_from_material_df(material_df, param)
        if not values:
            continue

        style = detect_data_param_style(raw, param)
        touched_indices = set()
        replaced_array = False
        new_lines: List[str] = []

        for line in lines:
            clean = _strip_inline_comment(line)
            parts = re.split(r"\s+", clean) if clean else []
            if not parts:
                new_lines.append(line)
                continue
            key0, idx0 = _line_key_and_index(parts[0])
            if key0.lower() != param.lower():
                new_lines.append(line)
                continue

            # Preserve trailing comments, if any.
            comment = ""
            if "#" in line:
                comment = "  #" + line.split("#", 1)[1]
            elif "//" in line:
                comment = "  //" + line.split("//", 1)[1]

            if style == "array" and not replaced_array:
                old_vals = parts[1:]
                start_idx = detect_data_param_array_start(raw, param, max_type=5)
                # Preserve compact d.data rows such as:
                #   YoungModulus E2 E3 E4 E5
                # by writing exactly the same number of columns unless the user
                # enabled additional types in the material table.
                if param in D_DATA_ARRAY_LAYOUTS:
                    layout_start, layout_count = D_DATA_ARRAY_LAYOUTS[param]
                    start_idx = layout_start
                    end_idx = layout_start + layout_count - 1
                else:
                    end_idx = max(start_idx + len(old_vals) - 1, max(values.keys()))
                arr = []
                for idx in range(start_idx, end_idx + 1):
                    old_pos = idx - start_idx
                    if idx in values:
                        arr.append(values[idx])
                    elif 0 <= old_pos < len(old_vals):
                        arr.append(old_vals[old_pos])
                    else:
                        arr.append("0")
                new_lines.append(f"{param} " + " ".join(arr) + comment)
                replaced_array = True
                touched_indices.update(range(start_idx, end_idx + 1))
                continue
            elif style == "indexed" and len(parts) >= 3 and parts[1].isdigit():
                idx = int(parts[1])
                if idx in values:
                    new_lines.append(f"{param} {idx} {values[idx]}" + comment)
                    touched_indices.add(idx)
                else:
                    new_lines.append(line)
                continue
            elif style == "bracket" and idx0 is not None:
                if idx0 in values:
                    new_lines.append(f"{param}[{idx0}] {values[idx0]}" + comment)
                    touched_indices.add(idx0)
                else:
                    new_lines.append(line)
                continue
            elif style == "suffix" and idx0 is not None:
                if idx0 in values:
                    new_lines.append(f"{param}{idx0} {values[idx0]}" + comment)
                    touched_indices.add(idx0)
                else:
                    new_lines.append(line)
                continue
            else:
                new_lines.append(line)

        if style == "missing":
            append_lines.append("")
            append_lines.append(f"# GUI material settings: {param}")
            if param in D_DATA_ARRAY_LAYOUTS:
                layout_start, layout_count = D_DATA_ARRAY_LAYOUTS[param]
                arr = []
                for idx in range(layout_start, layout_start + layout_count):
                    arr.append(values.get(idx, "0"))
                append_lines.append(f"{param}	" + "	".join(arr))
            else:
                for idx in sorted(values.keys()):
                    append_lines.append(f"{param} {idx} {values[idx]}")
        elif style in {"indexed", "bracket", "suffix"}:
            missing = sorted(set(values.keys()) - touched_indices)
            if missing:
                append_lines.append("")
                append_lines.append(f"# GUI material settings: added missing {param} entries")
                for idx in missing:
                    if style == "bracket":
                        append_lines.append(f"{param}[{idx}] {values[idx]}")
                    elif style == "suffix":
                        append_lines.append(f"{param}{idx} {values[idx]}")
                    else:
                        append_lines.append(f"{param} {idx} {values[idx]}")
        lines = new_lines

    text = "\n".join(lines)
    if append_lines:
        text = text.rstrip() + "\n" + "\n".join(append_lines) + "\n"
    else:
        text = text + ("\n" if raw.endswith("\n") else "")
    return text

def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def start_background_cmd(command: str, cwd: Path, log_path: Path, pid_key: str, name: str) -> None:
    if not cwd.exists():
        st.warning(f"⚠️ Working directory was not found: `{cwd}`")
        return
    if process_alive(st.session_state.get(pid_key)):
        notice(f"⚠️ {name} is already running. Stop it before starting again.", "yellow")
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["/bin/bash", "-lc", command],
        cwd=str(cwd),
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    f.close()
    st.session_state[pid_key] = proc.pid
    st.session_state[f"{pid_key}_log"] = str(log_path)
    notice(f"▶ Started {name}. PID=<code>{proc.pid}</code>", "green")

def stop_background_cmd(pid_key: str, name: str) -> None:
    pid = st.session_state.get(pid_key)
    if not process_alive(pid):
        notice(f"{name} is not running.", "yellow")
        st.session_state[pid_key] = None
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(0.5)
        if process_alive(pid):
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        st.session_state[pid_key] = None
        notice(f"⏹ Stopped {name}.", "yellow")
    except Exception as e:
        st.warning(f"Failed to stop: {e}")

def show_process_status(pid_key: str, name: str) -> None:
    pid = st.session_state.get(pid_key)
    alive = process_alive(pid)
    status = "🟢 Running" if alive else "⚪ Stopped"
    st.markdown(f'<div class="kill-box"><b>{name}</b><br>Status: {status}<br>PID: {pid if pid else "-"}</div>', unsafe_allow_html=True)
    log_path = st.session_state.get(f"{pid_key}_log")
    if log_path:
        lp = Path(log_path)
        if lp.exists():
            tail = "\n".join(lp.read_text(encoding="utf-8", errors="replace").splitlines()[-250:])
            st.code(tail if tail else "(log is empty)", language="bash")

def parse_points_from_vtk(path: Path, max_points: int = 200000) -> pd.DataFrame:
    txt = read_text(path, "")
    m = re.search(r"POINTS\s+(\d+)\s+\w+\s+(.*?)(?:\n[A-Z_]+\s|\Z)", txt, flags=re.S)
    if not m:
        return pd.DataFrame(columns=["x", "y", "z"])
    vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(2))
    pts = []
    for i in range(0, min(len(vals), max_points * 3), 3):
        if i + 2 < len(vals):
            pts.append((float(vals[i]), float(vals[i+1]), float(vals[i+2])))
    return pd.DataFrame(pts, columns=["x", "y", "z"])

def parse_points_generic(path: Path, max_points: int = 200000) -> pd.DataFrame:
    pts = []
    for line in read_text(path, "").splitlines():
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if len(nums) >= 3:
            try:
                pts.append((float(nums[0]), float(nums[1]), float(nums[2])))
            except Exception:
                pass
        if len(pts) >= max_points:
            break
    return pd.DataFrame(pts, columns=["x", "y", "z"])

def find_particle_preview_file(workdir: Path) -> Path | None:
    candidates = []
    for suffix in ["*.vtk", "*.prof", "*.grid", "*.dat", "*.txt"]:
        candidates += list(workdir.glob(suffix))
    candidates = [p for p in candidates if p.is_file() and p.stat().st_size > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def load_preview_points(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".vtk":
        df = parse_points_from_vtk(path)
        if len(df) > 0:
            return df
    return parse_points_generic(path)

def show_particle_preview(workdir: Path, dim: str) -> None:
    pf = find_particle_preview_file(workdir)
    if pf is None:
        notice("No previewable particle file was found.", "yellow")
        return
    df = load_preview_points(pf)
    if df.empty:
        notice(f"The file was found, but point coordinates could not be read: <code>{pf.name}</code>", "yellow")
        return
    if len(df) > 50000:
        df = df.sample(50000, random_state=1)
        notice("The point cloud is large, so it was downsampled to 50,000 points.", "yellow")
    st.markdown(f"**Preview file:** `{pf.name}`  ({len(df):,} points shown)")
    if dim == "2D":
        st.scatter_chart(df, x="x", y="y", height=520)
    else:
        try:
            import plotly.express as px
            fig = px.scatter_3d(df, x="x", y="y", z="z", opacity=0.55, height=650)
            fig.update_traces(marker=dict(size=2))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            notice("Plotly is unavailable; showing XY projection.", "yellow")
            st.scatter_chart(df, x="x", y="y", height=520)


def stat_cards(pairs: list) -> None:
    html = '<div class="stat-grid">'
    for label, val, ok in pairs:
        cls = "ok" if ok else "ng"
        html += f'<div class="stat-card {cls}"><div class="slabel">{label}</div><div class="sval">{val}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def step_bar(active: int) -> None:
    steps = [
        ("Generator", "Build particle generator"),
        ("Source", "Build solver"),
        ("Geometry", "Edit BOID"),
        (".data", "Edit materials"),
        ("Generate", "Create particles"),
        ("Execute", "Run solver"),
    ]
    html = '<div class="step-bar">'
    for i, (en, sub) in enumerate(steps, 1):
        cls = "active" if i == active else ""
        html += (f'<div class="step-item {cls}">'
                 f'<div class="step-num">{i}</div>'
                 f'<div class="step-label"><span class="en">{en}</span><span class="ja">{sub}</span></div>'
                 f'</div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# =============================================================================
# Path discovery
# =============================================================================
def get_script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()

def find_project_root_from(start: Path, max_up: int = 8) -> Path:
    cur = start.resolve()
    for _ in range(max_up + 1):
        if (cur / "source").exists() and (cur / "generator").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()

def to_rel(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())

def resolve_path(project_root: Path, user_value: str) -> Path:
    p = Path(user_value).expanduser()
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()

def find_case_dirs(project_root: Path) -> List[Path]:
    out: List[Path] = []
    for parent_name in ["results", "results2"]:
        parent = project_root / parent_name
        if not parent.exists():
            continue
        for child in sorted(parent.iterdir()):
            if child.is_dir() and child not in out:
                out.append(child)
    for p in [project_root / "results2" / "LandSlide", project_root / "results" / "LandSlide"]:
        if p.exists() and p not in out:
            out.insert(0, p)
    if not out:
        out = [project_root / "results" / "LandSlide"]
    return out

def list_files_with_suffix(case_dir: Path, suffix: str) -> List[Path]:
    if not case_dir.exists():
        return []
    return sorted([p for p in case_dir.iterdir() if p.is_file() and p.suffix.lower() == suffix.lower()])

def ensure_path_defaults(reset: bool = False) -> None:
    script_dir = get_script_dir()
    project_root = find_project_root_from(script_dir)
    case_dirs = find_case_dirs(project_root)
    default_case_dir = case_dirs[0]
    data_files = list_files_with_suffix(default_case_dir, ".data")
    boid_files = list_files_with_suffix(default_case_dir, ".boid")
    defaults = {
        "project_root_input": str(project_root),
        "workdir_input": to_rel(default_case_dir, project_root),
        "case_input": to_rel(data_files[0] if data_files else default_case_dir / "dem.data", project_root),
        "boid_input": to_rel(boid_files[0] if boid_files else default_case_dir / "particles.boid", project_root),
        "results_input": to_rel(default_case_dir, project_root),
        "generator_build_dir": "generator",
        "source_build_dir": "source",
        "generator_build_cmd": "make",
        "source_build_cmd": "make",
        "generator_clean_cmd": "make clean",
        "source_clean_cmd": "make clean",
        "generator_cmd": "../../generator/MK-SPH DEM",
        "execute_cmd": "../../source/MK-SPH dem.data DEM.grid Dam%03d.prof granularImplicit%03d.vtk box.log 4",
        "source_makefile_relpath": "source/makefile",
    }
    for k, v in defaults.items():
        if reset or k not in st.session_state:
            st.session_state[k] = v

# =============================================================================
# Shell IO
# =============================================================================
def read_text(path: Path, fallback: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else fallback
    except UnicodeDecodeError:
        return path.read_text(encoding="cp932", errors="replace") if path.exists() else fallback
    except Exception:
        return fallback

def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def run_cmd(command: str, cwd: Path, title: str) -> int:
    """Run shell command, stream output, never raise. Returns exit code."""
    if not cwd.exists():
        st.warning(f"⚠️ Working directory was not found: `{cwd}`")
        notice("Please check the path. You can still move to the next step after this error.", "yellow")
        return 127
    try:
        proc = subprocess.Popen(
            ["/bin/bash", "-lc", command], cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        lines: List[str] = []
        placeholder = st.empty()
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line.rstrip("\n"))
            placeholder.code("\n".join(lines[-300:]), language="bash")
        code = proc.wait()
        placeholder.code("\n".join(lines[-300:]), language="bash")
        st.session_state["last_command_log"] = "\n".join(lines)
        return code
    except Exception as e:
        st.warning(f"⚠️ Command execution failed: {e}")
        return -1

def run_with_result(command: str, cwd: Path, label: str, ok_msg: str, ng_msg: str) -> int:
    """Run command and show colored result. Never blocks."""
    code = run_cmd(command, cwd, label)
    if code == 0:
        notice(f"✅ Completed — {ok_msg}", "green")
    else:
        notice(
            f"⚠️ Warning or error (exit code {code}) — {ng_msg}<br>"
            f"Check the log above. <b>You can still move to the next step.</b>",
            "yellow"
        )
    return code

def _script_text_or_default(text: str | None, default_command: str) -> str:
    """Return a safe shell script body; never return an empty script.

    Some case folders may not have generate.sh/execute.sh yet, or an existing
    file may be accidentally empty.  In those cases the GUI must not create an
    empty script, because the Run button would silently do nothing.
    """
    body = (text or "").strip()
    if not body:
        body = (default_command or "").strip()
    if not body:
        body = "echo 'ERROR: command is empty. Please set this command in the GUI.'\nexit 1"
    if not body.startswith("#!"):
        body = "#!/bin/bash\nset -e\n" + body
    return body.rstrip() + "\n"

def default_generate_script_text() -> str:
    return _script_text_or_default(None, st.session_state.get("generator_cmd", "../../generator/MK-SPH DEM"))

def default_execute_script_text() -> str:
    return _script_text_or_default(None, st.session_state.get("execute_cmd", "../../source/MK-SPH dem.data DEM.grid Dam%03d.prof granularImplicit%03d.vtk box.log 4"))

def sync_sh_from_disk(generate_sh: Path, execute_sh: Path) -> None:
    current = (str(generate_sh), str(execute_sh))
    if st.session_state.get("_loaded_sh") != current:
        gen_text = read_text(generate_sh, "")
        exe_text = read_text(execute_sh, "")
        st.session_state["generate_sh_text"] = _script_text_or_default(
            gen_text, st.session_state.get("generator_cmd", "../../generator/MK-SPH DEM")
        )
        st.session_state["execute_sh_text"] = _script_text_or_default(
            exe_text, st.session_state.get("execute_cmd", "../../source/MK-SPH dem.data DEM.grid Dam%03d.prof granularImplicit%03d.vtk box.log 4")
        )
        st.session_state["_loaded_sh"] = current

def save_generate_sh(generate_sh: Path) -> None:
    # Do NOT assign to st.session_state["generate_sh_text"] here.
    # Streamlit forbids modifying a widget-backed session_state key
    # after the text_area widget has been instantiated in the same run.
    text = _script_text_or_default(
        st.session_state.get("generate_sh_text", ""),
        st.session_state.get("generator_cmd", "../../generator/MK-SPH DEM"),
    )
    write_text(generate_sh, text)
    generate_sh.chmod(0o755)

def save_execute_sh(execute_sh: Path) -> None:
    # Do NOT assign to st.session_state["execute_sh_text"] here.
    # This avoids: StreamlitAPIException: cannot be modified after widget instantiated.
    text = _script_text_or_default(
        st.session_state.get("execute_sh_text", ""),
        st.session_state.get("execute_cmd", "../../source/MK-SPH dem.data DEM.grid Dam%03d.prof granularImplicit%03d.vtk box.log 4"),
    )
    write_text(execute_sh, text)
    execute_sh.chmod(0o755)

def save_sh(generate_sh: Path, execute_sh: Path) -> None:
    """Backward-compatible wrapper. Never writes empty scripts."""
    save_generate_sh(generate_sh)
    save_execute_sh(execute_sh)

# =============================================================================
# Makefile
# =============================================================================
GPU_PRESETS = {
    "A100 (cc80)": "80",
    "RTX 6000 Ada (cc89)": "89",
    "RTX PRO 6000 Blackwell (cc120)": "120",
    "Custom": "",
}


def discover_hpcsdk_versions(module_family_dir: str) -> List[str]:
    """Return available NVHPC module versions under a module family directory.

    Preferred server-independent style:
      module use /opt/nvidia/hpc_sdk/modulefiles/nvhpc
      module load 25.11

    The function also tolerates the older root style
    /opt/nvidia/hpc_sdk/modulefiles and returns names like nvhpc/25.11.
    """
    root = Path(module_family_dir).expanduser()
    versions: List[str] = []
    if not root.exists():
        return versions

    # New default: /opt/nvidia/hpc_sdk/modulefiles/nvhpc contains version files.
    for child in sorted(root.iterdir()):
        if child.is_file():
            versions.append(child.name)

    # Backward-compatible root: /opt/nvidia/hpc_sdk/modulefiles/nvhpc/25.11.
    for family in sorted(root.iterdir()):
        if not family.is_dir():
            continue
        for version_file in sorted(family.iterdir()):
            if version_file.is_file():
                versions.append(f"{family.name}/{version_file.name}")

    return sorted(set(versions), reverse=True)


def normalize_hpcsdk_module_load_name(module_family_dir: str, version: str) -> str:
    """Build the module name to load from GUI fields.

    If the module directory is the family directory ending in /nvhpc, load only
    the version, e.g. 25.11. If the user gives the root modulefiles directory,
    load nvhpc/<version>. If the version already contains '/', preserve it.
    """
    module_dir = Path(module_family_dir).expanduser()
    v = (version or "").strip()
    if not v:
        return ""
    if "/" in v:
        return v
    if module_dir.name == "nvhpc":
        return v
    return f"nvhpc/{v}"


def discover_nvcpp_paths() -> List[str]:
    """Search common HPCSDK locations for nvc++."""
    candidates: List[str] = []
    roots = [Path("/opt/nvidia/hpc_sdk"), Path.home() / "nvidia" / "hpc_sdk"]
    for root in roots:
        if not root.exists():
            continue
        candidates.extend(str(p) for p in root.glob("**/compilers/bin/nvc++") if p.is_file())
    # Prefer newer-looking paths last alphabetically first in the displayed list.
    return sorted(set(candidates), reverse=True)


def hpcsdk_env_prefix() -> str:
    """Shell prefix to make the module command and nvc++ available during GUI builds."""
    if not st.session_state.get("mf_use_hpcsdk_module", False):
        return ""
    module_dir = st.session_state.get("mf_hpcsdk_module_dir", "").strip()
    module_version = st.session_state.get("mf_hpcsdk_module_version", "").strip()
    # Backward compatibility with older GUI state.
    if not module_version:
        module_version = st.session_state.get("mf_hpcsdk_module_name", "").strip()
    module_load_name = normalize_hpcsdk_module_load_name(module_dir, module_version)
    if not module_dir or not module_load_name:
        return ""
    # Keep this POSIX-shell friendly. The command is executed by /bin/bash -lc.
    return (
        "if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi; "
        f"module use {module_dir}; module load {module_load_name}; "
    )


def with_hpcsdk_env(command: str) -> str:
    """Add HPCSDK module setup before a build command when OpenACC mode is selected.

    If the user already wrote a module command into Source build command,
    do not duplicate it.
    """
    if "GPU" in st.session_state.get("mf_mode", ""):
        if "module load" in command or "module use" in command:
            return command
        return hpcsdk_env_prefix() + command
    return command

def init_makefile_state(force: bool = False) -> None:
    defaults = {
        "mf_target": "MK-SPH",
        "mf_mode": "CPU — Multi-core (OpenMP)",
        "mf_cc_cpu": "g++-15",
        "mf_cc_gpu": "nvc++",
        "mf_use_debug": False,
        "mf_use_wall": False,
        "mf_extra_cflags": "",
        "mf_extra_ldflags": "",
        "mf_wflags": "",
        "mf_objects": "main_Implicit.o errorfunc.o log.o",
        "mf_ldflags_base": "-lm",
        "mf_gpu_preset": "RTX 6000 Ada (cc89)",
        "mf_gpu_cc": "89",
        "mf_use_hpcsdk_module": True,
        "mf_hpcsdk_module_dir": "/opt/nvidia/hpc_sdk/modulefiles/nvhpc",
        "mf_hpcsdk_module_version": "25.11",
        "mf_hpcsdk_module_name": "25.11",  # backward-compatible alias
        "mf_use_direct_nvcpp_path": False,
        "mf_nvcpp_path": "/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nvc++",
        "mf_backup_before_save": True,
        "mf_clean_rule": "/bin/rm -f *.o *~ $(TARGET) $(TARGET).exe",
        "makefile_editor_text": "",
    }
    for k, v in defaults.items():
        if force or k not in st.session_state:
            st.session_state[k] = v

def build_makefile_text() -> str:
    mode = st.session_state["mf_mode"]
    if "Single" in mode or "1 core" in mode:
        cc = st.session_state["mf_cc_cpu"].strip() or "g++-15"
        cflags = "-O3"
    elif "OpenMP" in mode or "Multi" in mode:
        cc = st.session_state["mf_cc_cpu"].strip() or "g++-15"
        cflags = "-O3 -fopenmp"
    else:  # GPU
        if st.session_state.get("mf_use_direct_nvcpp_path") and st.session_state.get("mf_nvcpp_path", "").strip():
            cc = st.session_state["mf_nvcpp_path"].strip()
        else:
            cc = st.session_state["mf_cc_gpu"].strip() or "nvc++"
        cc_num = st.session_state["mf_gpu_cc"].strip() or "89"
        cflags = f"-acc -O3 -Minfo=accel -gpu=cc{cc_num} -cuda"
    if st.session_state["mf_use_debug"]:
        cflags += " -g"
    if st.session_state["mf_use_wall"]:
        cflags += " -Wall"
    if st.session_state["mf_extra_cflags"].strip():
        cflags += " " + st.session_state["mf_extra_cflags"].strip()
    ldflags = st.session_state["mf_ldflags_base"].strip() or "-lm"
    if st.session_state["mf_extra_ldflags"].strip():
        ldflags += " " + st.session_state["mf_extra_ldflags"].strip()
    wflags = st.session_state["mf_wflags"].strip()
    lines = [
        f"TARGET = {st.session_state['mf_target'].strip() or 'MK-SPH'}", "",
        f"CC = {cc}",
        f"OBJE = {st.session_state['mf_objects'].strip() or 'main_Implicit.o errorfunc.o log.o'}", "",
        f"CFLAGS = {cflags}",
        f"LDFLAGS = {ldflags}",
        f"WFLAGS = {wflags}" if wflags else "WFLAGS =", "",
        "$(TARGET): $(OBJE)",
        "\t$(CC) $(CFLAGS) $(WFLAGS) -o $@ $(OBJE) $(LDFLAGS)", "",
        ".cpp.o:",
        "\t$(CC) $(CFLAGS) $(WFLAGS) -c $<", "",
        "clean:",
        f"\t{st.session_state['mf_clean_rule'].strip() or '/bin/rm -f *.o *~ $(TARGET) $(TARGET).exe'}", "",
        "main_Implicit.o\t\t: log.h",
        "errorfunc.o\t\t: errorfunc.h",
        "log.o\t\t\t: errorfunc.h log.h", "",
    ]
    return "\n".join(lines)

def backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.name + f".bak_{ts}")
    if path.exists():
        shutil.copy2(path, backup)
    return backup

# =============================================================================
# BOID
# =============================================================================
BOID_COLUMNS = [
    "Name", "Kind", "Mode", "Type", "RigidType",
    "XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax",
    "CenterX", "CenterY", "CenterZ",
    "Radius", "Height", "AngleDeg",
    "VelocityX", "VelocityY", "VelocityZ", "Enthalpy"
]

def default_boid_state() -> None:
    """Initialize BOID editor state.

    generator.cpp-compatible rule:
      2D domain: z = [0.0, ParticleDistance]
      2D shape output: ZLayer ParticleDistance
      3D shape output: ZRange ZMin ZMax
    """
    st.session_state.setdefault("boid_dim", "2D")
    st.session_state.setdefault("boid_particle_distance", 0.001)
    st.session_state.setdefault("boid_lower_x", -0.1)
    st.session_state.setdefault("boid_lower_y", 0.0)
    st.session_state.setdefault("boid_lower_z", 0.0)
    st.session_state.setdefault("boid_upper_x", 0.21)
    st.session_state.setdefault("boid_upper_y", 0.17)
    st.session_state.setdefault("boid_upper_z", float(st.session_state.get("boid_particle_distance", 0.001)))

    if "boid_shapes" not in st.session_state:
        pdist = float(st.session_state.get("boid_particle_distance", 0.001))
        z0, z1, zc = 0.0, pdist, 0.5 * pdist
        st.session_state["boid_shapes"] = pd.DataFrame([
            ["elastoplastic", "box", "add", 2, -1, 0.0, 0.1, 0.003, 0.053, z0, z1, 0.05, 0.028, zc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ["floor",        "box", "add", 4, -1, 0.0, 0.2, 0.0,   0.003, z0, z1, 0.1,  0.0015, zc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ["right_wall",   "box", "add", 4, -1, 0.2, 0.203, 0.0, 0.17,  z0, z1, 0.2015, 0.085, zc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ["left_wall",    "box", "add", 4, -1, -0.003, 0.0, 0.0, 0.17, z0, z1, -0.0015, 0.085, zc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ], columns=BOID_COLUMNS).astype({"Type": "int64", "RigidType": "int64"})


def _float_after(pattern: str, text: str, default: float) -> float:
    m = re.search(pattern, text, flags=re.I)
    return float(m.group(1)) if m else default


def _token_from_block(block: str, key: str, default: str = "") -> str:
    mm = re.search(rf"^\s*{re.escape(key)}\s+(.+?)\s*$", block, flags=re.I | re.M)
    return mm.group(1).strip() if mm else default


def _as_float_list(text: str, n: int, default: List[float]) -> List[float]:
    try:
        vals = [float(x) for x in text.split()[:n]]
        if len(vals) == n:
            return vals
    except Exception:
        pass
    return default


def _normalize_kind(kind: str) -> str:
    kind = (kind or "box").strip().lower()
    return {
        "cuboid": "box", "rectangle": "box", "rect": "box",
        "cyl": "cylinder",
        "tri": "triangle", "prism": "triangle", "triangular_prism": "triangle",
    }.get(kind, kind)


def _shape_row_from_new_block(block: str, idx: int, pdist: float) -> List:
    name = _token_from_block(block, "Name", f"shape_{idx}")
    kind = _normalize_kind(_token_from_block(block, "Kind", "box"))
    mode = _token_from_block(block, "Mode", "add").lower()
    typ = int(float(_token_from_block(block, "Type", "2")))
    rigid = int(float(_token_from_block(block, "RigidType", "-1")))

    xr = _as_float_list(_token_from_block(block, "XRange", ""), 2, [0.0, 0.0])
    yr = _as_float_list(_token_from_block(block, "YRange", ""), 2, [0.0, 0.0])

    zr_txt = _token_from_block(block, "ZRange", "")
    zl_txt = _token_from_block(block, "ZLayer", "")
    if zr_txt:
        zr = _as_float_list(zr_txt, 2, [0.0, pdist])
    elif zl_txt:
        # ZLayer is the 2D layer thickness in generator.cpp.
        zr = [0.0, float(zl_txt.split()[0])]
    else:
        zr = [0.0, pdist]

    center = [0.5 * (xr[0] + xr[1]), 0.5 * (yr[0] + yr[1]), 0.5 * (zr[0] + zr[1])]
    if _token_from_block(block, "Center", ""):
        center = _as_float_list(_token_from_block(block, "Center", ""), 3, center)

    radius = float(_token_from_block(block, "Radius", "0").split()[0]) if _token_from_block(block, "Radius", "") else 0.0
    height = float(_token_from_block(block, "Height", "0").split()[0]) if _token_from_block(block, "Height", "") else 0.0
    angle = float(_token_from_block(block, "AngleDeg", "0").split()[0]) if _token_from_block(block, "AngleDeg", "") else 0.0
    vel = _as_float_list(_token_from_block(block, "Velocity", ""), 3, [0.0, 0.0, 0.0])
    enthalpy = float(_token_from_block(block, "Enthalpy", "0").split()[0]) if _token_from_block(block, "Enthalpy", "") else 0.0

    if kind != "box" and xr == [0.0, 0.0] and yr == [0.0, 0.0] and radius > 0.0:
        xr = [center[0] - radius, center[0] + radius]
        yr = [center[1] - radius, center[1] + radius]

    return [name, kind, mode, typ, rigid, xr[0], xr[1], yr[0], yr[1], zr[0], zr[1],
            center[0], center[1], center[2], radius, height, angle, vel[0], vel[1], vel[2], enthalpy]


def _shape_row_from_start_cuboid(block: str, idx: int, pdist: float) -> List:
    spacing = float(_token_from_block(block, "Spacing", str(pdist)).split()[0]) if _token_from_block(block, "Spacing", "") else pdist
    typ = int(float(_token_from_block(block, "Type", "2")))
    rigid = int(float(_token_from_block(block, "RigidType", "-1")))
    lower = _as_float_list(_token_from_block(block, "Lower", ""), 3, [0.0, 0.0, 0.0])
    upper = _as_float_list(_token_from_block(block, "Upper", ""), 3, [0.0, 0.0, spacing])
    vel = _as_float_list(_token_from_block(block, "Velocity", ""), 3, [0.0, 0.0, 0.0])
    enthalpy = float(_token_from_block(block, "Enthalpy", "0").split()[0]) if _token_from_block(block, "Enthalpy", "") else 0.0
    name = _token_from_block(block, "Name", f"cuboid_{idx}")
    return [name, "box", "add", typ, rigid, lower[0], upper[0], lower[1], upper[1], lower[2], upper[2],
            0.5 * (lower[0] + upper[0]), 0.5 * (lower[1] + upper[1]), 0.5 * (lower[2] + upper[2]),
            0.0, 0.0, 0.0, vel[0], vel[1], vel[2], enthalpy]


def parse_boid_text(raw: str) -> None:
    """Accept Shape/StartShape and old StartCuboid formats."""
    default_boid_state()

    st.session_state["boid_particle_distance"] = _float_after(
        r"ParticleDistance\s+([-+0-9.eE]+)", raw, st.session_state["boid_particle_distance"]
    )
    pdist = float(st.session_state["boid_particle_distance"])

    m = re.search(r"LowerDomain\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", raw, flags=re.I)
    if m:
        st.session_state["boid_lower_x"], st.session_state["boid_lower_y"], st.session_state["boid_lower_z"] = map(float, m.groups())
    m = re.search(r"UpperDomain\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", raw, flags=re.I)
    if m:
        st.session_state["boid_upper_x"], st.session_state["boid_upper_y"], st.session_state["boid_upper_z"] = map(float, m.groups())

    if abs(float(st.session_state["boid_upper_z"]) - float(st.session_state["boid_lower_z"])) <= 1.5 * pdist:
        st.session_state["boid_dim"] = "2D"
    else:
        st.session_state["boid_dim"] = "3D"

    rows = []
    for i, block in enumerate(re.findall(r"^\s*(?:StartShape|Shape)\s*(.*?)^\s*EndShape\s*$", raw, flags=re.I | re.S | re.M), 1):
        rows.append(_shape_row_from_new_block(block, i, pdist))

    for i, block in enumerate(re.findall(r"^\s*StartCuboid\s*(.*?)^\s*EndCuboid\s*$", raw, flags=re.I | re.S | re.M), 1):
        rows.append(_shape_row_from_start_cuboid(block, i, pdist))

    if rows:
        df = pd.DataFrame(rows, columns=BOID_COLUMNS)
        df["Type"] = df["Type"].fillna(2).astype("int64")
        df["RigidType"] = df["RigidType"].fillna(-1).astype("int64")
        st.session_state["boid_shapes"] = df


def _fmt(x) -> str:
    return f"{float(x):g}"


def render_boid_text() -> str:
    """Write .boid in generator.cpp-compatible format.

    2D writes ZLayer. 3D writes ZRange.
    """
    pdist = float(st.session_state["boid_particle_distance"])
    dim = st.session_state["boid_dim"]

    lower_z = 0.0 if dim == "2D" else float(st.session_state["boid_lower_z"])
    upper_z = pdist if dim == "2D" else float(st.session_state["boid_upper_z"])
    zcenter2d = 0.5 * pdist

    lines = [
        "# BOID geometry file generated by Granular_MPH GUI",
        "# Compatible with generator.cpp Shape parser",
        f"ParticleDistance {pdist:g}",
        f"LowerDomain {_fmt(st.session_state['boid_lower_x'])} {_fmt(st.session_state['boid_lower_y'])} {lower_z:g}",
        f"UpperDomain {_fmt(st.session_state['boid_upper_x'])} {_fmt(st.session_state['boid_upper_y'])} {upper_z:g}",
        "",
    ]

    df = st.session_state["boid_shapes"].fillna("")
    for _, r in df.iterrows():
        kind = _normalize_kind(str(r.get("Kind", "box")))
        mode = str(r.get("Mode", "add")).lower().strip() or "add"
        typ_val = int(float(r.get("Type", 2) or 2))
        rigid_val = int(float(r.get("RigidType", -1) or -1))

        lines += [
            "Shape",
            f"  Name {r.get('Name', 'shape')}",
            f"  Kind {kind}",
            f"  Mode {mode}",
            f"  Type {typ_val}",
            f"  RigidType {rigid_val}",
        ]

        if kind == "box":
            lines += [
                f"  XRange {_fmt(r['XMin'])} {_fmt(r['XMax'])}",
                f"  YRange {_fmt(r['YMin'])} {_fmt(r['YMax'])}",
            ]
        elif kind in ["circle", "sphere", "cylinder"]:
            cz = zcenter2d if dim == "2D" else float(r["CenterZ"])
            lines += [
                f"  Center {_fmt(r['CenterX'])} {_fmt(r['CenterY'])} {cz:g}",
                f"  Radius {_fmt(r['Radius'])}",
            ]
        elif kind == "triangle":
            cz = zcenter2d if dim == "2D" else float(r["CenterZ"])
            lines += [
                f"  Center {_fmt(r['CenterX'])} {_fmt(r['CenterY'])} {cz:g}",
                f"  Radius {_fmt(r['Radius'])}",
                f"  AngleDeg {_fmt(r['AngleDeg'])}",
            ]
        else:
            cz = zcenter2d if dim == "2D" else float(r["CenterZ"])
            lines += [
                f"  Center {_fmt(r['CenterX'])} {_fmt(r['CenterY'])} {cz:g}",
                f"  Radius {_fmt(r['Radius'])}",
            ]

        if dim == "2D":
            lines.append(f"  ZLayer {pdist:g}")
        else:
            lines.append(f"  ZRange {_fmt(r['ZMin'])} {_fmt(r['ZMax'])}")
            if kind in ["cylinder", "triangle"]:
                lines.append(f"  Height {_fmt(r['Height'])}")

        lines += [
            f"  Velocity {_fmt(r['VelocityX'])} {_fmt(r['VelocityY'])} {_fmt(r['VelocityZ'])}",
            f"  Enthalpy {_fmt(r['Enthalpy'])}",
            "EndShape",
            "",
        ]

    return "\n".join(lines)


def list_result_files(results_dir: Path) -> pd.DataFrame:
    rows = []
    if not results_dir.exists():
        return pd.DataFrame(columns=["File name", "Kind", "Size (bytes)"])
    for p in sorted(results_dir.iterdir()):
        rows.append({
            "File name": p.name,
            "Kind": "📁 Folder" if p.is_dir() else p.suffix or "(no extension)",
            "Size (bytes)": "" if p.is_dir() else f"{p.stat().st_size:,}"
        })
    return pd.DataFrame(rows)


def list_recent_output_files(results_dir: Path, limit: int = 20) -> List[Path]:
    """Return recent solver/generator output files for quick inspection/download."""
    if not results_dir.exists():
        return []
    suffixes = {".vtk", ".prof", ".grid", ".log", ".dat", ".csv", ".txt"}
    files = [p for p in results_dir.iterdir() if p.is_file() and (p.suffix.lower() in suffixes or p.name.startswith("box"))]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]

def validate_boid_shapes(df: pd.DataFrame, dim: str) -> List[str]:
    """Lightweight GUI-side checks before writing/generating particles."""
    warnings: List[str] = []
    if df is None or df.empty:
        return ["No shapes are defined."]
    valid_kinds = {"box", "circle", "cylinder", "sphere", "triangle"}
    valid_modes = {"add", "remove"}
    for idx, r in df.fillna("").iterrows():
        name = str(r.get("Name", f"row_{idx}")) or f"row_{idx}"
        kind = _normalize_kind(str(r.get("Kind", "box")))
        mode = str(r.get("Mode", "add")).lower().strip()
        if kind not in valid_kinds:
            warnings.append(f"{name}: unknown Kind '{kind}'.")
        if mode not in valid_modes:
            warnings.append(f"{name}: Mode should be add or remove.")
        try:
            if kind == "box":
                if float(r["XMax"]) <= float(r["XMin"]):
                    warnings.append(f"{name}: XMax must be larger than XMin.")
                if float(r["YMax"]) <= float(r["YMin"]):
                    warnings.append(f"{name}: YMax must be larger than YMin.")
                if dim == "3D" and float(r["ZMax"]) <= float(r["ZMin"]):
                    warnings.append(f"{name}: ZMax must be larger than ZMin in 3D.")
            else:
                if float(r.get("Radius", 0.0)) <= 0.0:
                    warnings.append(f"{name}: Radius should be positive for {kind}.")
        except Exception:
            warnings.append(f"{name}: some numeric fields are invalid.")
    return warnings

def show_validation_box(title: str, messages: List[str]) -> None:
    if not messages:
        notice(f"✅ {title}: no obvious GUI-side problems were detected.", "green")
    else:
        msg = "<br>".join(f"• {m}" for m in messages[:20])
        if len(messages) > 20:
            msg += f"<br>• ... and {len(messages)-20} more"
        notice(f"⚠️ <b>{title}</b><br>{msg}", "yellow")

# =============================================================================
# Initialize
# =============================================================================
ensure_path_defaults()
init_makefile_state()
default_boid_state()

project_root = Path(st.session_state["project_root_input"]).expanduser().resolve()
workdir = resolve_path(project_root, st.session_state["workdir_input"])
case_file = resolve_path(project_root, st.session_state["case_input"])
boid_file = resolve_path(project_root, st.session_state["boid_input"])
results_dir = resolve_path(project_root, st.session_state["results_input"])
generator_build_path = resolve_path(project_root, st.session_state["generator_build_dir"])
source_build_path = resolve_path(project_root, st.session_state["source_build_dir"])
source_makefile_path = resolve_path(project_root, st.session_state["source_makefile_relpath"])
generate_sh = workdir / "generate.sh"
execute_sh = workdir / "execute.sh"
sync_sh_from_disk(generate_sh, execute_sh)
st.session_state.setdefault("raw_case_text", read_text(case_file, ""))
st.session_state.setdefault("raw_boid_text", read_text(boid_file, render_boid_text()))
# IMPORTANT:
# raw_boid_text is an internal value.
# boid_editor_text is the Streamlit text_area widget value.
# Do not use raw_boid_text directly as a widget key, otherwise Streamlit raises:
# "st.session_state.raw_boid_text cannot be modified after the widget ... is instantiated."
if "boid_editor_text" not in st.session_state:
    st.session_state["boid_editor_text"] = st.session_state["raw_boid_text"]
if "boid_loaded_path" not in st.session_state:
    parse_boid_text(st.session_state["raw_boid_text"])
    st.session_state["boid_loaded_path"] = str(boid_file)

# Reload file-backed widget state when the sidebar path is changed manually.
# This prevents editing one case while accidentally viewing text from another case.
if st.session_state.get("case_loaded_path") != str(case_file):
    st.session_state["raw_case_text"] = read_text(case_file, "")
    st.session_state["data_material_table"] = build_material_table_from_data(st.session_state["raw_case_text"])
    st.session_state["data_global_table"] = build_global_table_from_data(st.session_state["raw_case_text"])
    st.session_state["case_loaded_path"] = str(case_file)
if st.session_state.get("boid_loaded_path") != str(boid_file):
    st.session_state["raw_boid_text"] = read_text(boid_file, render_boid_text())
    st.session_state["boid_editor_text"] = st.session_state["raw_boid_text"]
    parse_boid_text(st.session_state["raw_boid_text"])
    st.session_state["boid_loaded_path"] = str(boid_file)

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Path & command settings")
    st.markdown('<div class="help-text">Project root directory containing generator/ and source/.</div>', unsafe_allow_html=True)
    st.text_input("Project root", key="project_root_input")

    project_root = Path(st.session_state["project_root_input"]).expanduser().resolve()
    case_dirs = find_case_dirs(project_root)
    labels = [to_rel(p, project_root) for p in case_dirs]
    idx = labels.index(st.session_state["workdir_input"]) if st.session_state["workdir_input"] in labels else 0

    st.markdown('<div class="help-text">Select a case folder under results/ or results2/.</div>', unsafe_allow_html=True)
    chosen_case = st.selectbox("Case folder", labels, index=idx)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply", use_container_width=True, type="primary"):
            selected = resolve_path(project_root, chosen_case)
            data_files = list_files_with_suffix(selected, ".data")
            boid_files = list_files_with_suffix(selected, ".boid")
            st.session_state["workdir_input"] = chosen_case
            st.session_state["results_input"] = chosen_case
            st.session_state["case_input"] = to_rel(data_files[0] if data_files else selected / "dem.data", project_root)
            st.session_state["boid_input"] = to_rel(boid_files[0] if boid_files else selected / "particles.boid", project_root)
            st.session_state["_loaded_sh"] = None
            st.session_state.pop("raw_case_text", None)
            st.session_state.pop("boid_loaded_path", None)
            st.rerun()
    with col2:
        if st.button("Rescan", use_container_width=True):
            ensure_path_defaults(reset=True)
            st.rerun()

    st.divider()
    st.markdown("**Path settings**")
    st.text_input("Working directory", key="workdir_input")
    st.text_input(".data file", key="case_input")
    st.text_input(".boid file", key="boid_input")
    st.text_input("Output directory", key="results_input")
    st.divider()
    st.markdown("**Build settings**")
    st.text_input("Generator build directory", key="generator_build_dir")
    st.text_input("Source build directory", key="source_build_dir")
    st.text_input("Source Makefile path", key="source_makefile_relpath")
    st.text_input("Generator build command", key="generator_build_cmd")
    st.text_input("Generator clean command", key="generator_clean_cmd")
    st.text_input("Source build command", key="source_build_cmd")
    st.text_input("Source clean command", key="source_clean_cmd")
    st.divider()
    st.markdown("**Run commands**")
    st.text_input("Particle generation command", key="generator_cmd")
    st.text_input("Solver command", key="execute_cmd")

# recompute paths after sidebar
project_root = Path(st.session_state["project_root_input"]).expanduser().resolve()
workdir = resolve_path(project_root, st.session_state["workdir_input"])
case_file = resolve_path(project_root, st.session_state["case_input"])
boid_file = resolve_path(project_root, st.session_state["boid_input"])
results_dir = resolve_path(project_root, st.session_state["results_input"])
generator_build_path = resolve_path(project_root, st.session_state["generator_build_dir"])
source_build_path = resolve_path(project_root, st.session_state["source_build_dir"])
source_makefile_path = resolve_path(project_root, st.session_state["source_makefile_relpath"])
generate_sh = workdir / "generate.sh"
execute_sh = workdir / "execute.sh"
sync_sh_from_disk(generate_sh, execute_sh)

# =============================================================================
# Hero
# =============================================================================
st.markdown("""
<div class="hero">
  <h1>Granular_MPH GUI v5</h1>
  <p class="sub">
    <span class="badge">English UI</span>
    <span class="badge">Robust v2</span>
    Particle method simulation workflow dashboard with geometry, material, build, generation, solver controls, and output inspection
  </p>
</div>
""", unsafe_allow_html=True)

# Current case status
stat_cards([
    ("Project root", str(project_root), project_root.exists()),
    ("Working directory", str(workdir), workdir.exists()),
    (".data file", str(case_file), case_file.exists()),
    (".boid file", str(boid_file), boid_file.exists()),
    ("source Makefile", str(source_makefile_path), source_makefile_path.exists()),
])

notice(
    "⚠️ <b>You can move to the next step even if a warning or error appears.</b> "
    "Check the log, fix the issue if necessary, and rerun the step.",
    "yellow"
)


# =============================================================================
# Exact .data format editor for MK-SPH d.data
# =============================================================================
# The user's .data file is a fixed line-oriented format.  Do not rewrite it into
# another style.  These helpers parse the exact keys and render the exact key
# order/row lengths shown by the user.
EXACT_DDATA_ROWS = [
    ("Dt", "scalar", 1, "Global time step"),
    ("ElasticDt", "scalar", 1, "Elastic/plastic time step"),
    ("OutputInterval", "scalar", 1, "Profile output interval"),
    ("VtkOutputInterval", "scalar", 1, "VTK output interval"),
    ("EndTime", "scalar", 1, "End time"),
    ("RadiusRatioA", "scalar", 1, "Radius ratio A"),
    ("RadiusRatioP", "scalar", 1, "Radius ratio P"),
    ("RadiusRatioV", "scalar", 1, "Radius ratio V"),
    ("Density", "type0_5", 6, "Type0 Type1 Type2 Type3 Type4 Type5"),
    ("BulkModulus", "type0_5", 6, "Type0 Type1 Type2 Type3 Type4 Type5"),
    ("BulkViscosity", "type0_5", 6, "Type0 Type1 Type2 Type3 Type4 Type5"),
    ("ShearViscosity", "type0_5", 6, "Type0 Type1 Type2 Type3 Type4 Type5"),
    ("SurfaceTension", "type0_3", 4, "Type0 Type1 Type2 Type3"),
    ("YoungModulus", "type2_5", 4, "Type2 Type3 Type4 Type5"),
    ("PoissonRatio", "type2_5", 4, "Type2 Type3 Type4 Type5"),
    ("Cohesion", "type2_5", 4, "Type2 Type3 Type4 Type5"),
    ("InternalFrictionAngle", "type2_5", 4, "Type2 Type3 Type4 Type5"),
    ("DilatancyFrictionAngle", "type2_5", 4, "Type2 Type3 Type4 Type5"),
    ("ActualDebrisSize", "type2_3", 2, "Type2 Type3"),
    ("InteractionRatio(Type0)", "interaction", 6, "Type0 Type1 Type2 Type3 Type4 Type5"),
    ("InteractionRatio(Type1)", "interaction", 6, "Type0 Type1 Type2 Type3 Type4 Type5"),
    ("InteractionRatio(Type2)", "interaction", 6, "Type0 Type1 Type2 Type3 Type4 Type5"),
    ("InteractionRatio(Type3)", "interaction", 6, "Type0 Type1 Type2 Type3 Type4 Type5"),
    ("InteractionRatio(Type4)", "interaction", 6, "Type0 Type1 Type2 Type3 Type4 Type5"),
    ("InteractionRatio(Type5)", "interaction", 6, "Type0 Type1 Type2 Type3 Type4 Type5"),
    ("Gravity", "vector", 1, "x y z"),
    ("Wall2", "wall", 1, "Center x y z    Velocity vx vy vz    Omega ox oy oz"),
    ("Wall3", "wall", 1, "Center x y z    Velocity vx vy vz    Omega ox oy oz"),
]

EXACT_DDATA_DEFAULTS = {
    "Dt": "1.0e-4",
    "ElasticDt": "1.0e-4",
    "OutputInterval": "1.0",
    "VtkOutputInterval": "1.0e-2",
    "EndTime": "2.0e1",
    "RadiusRatioA": "2.5",
    "RadiusRatioP": "2.5",
    "RadiusRatioV": "2.5",
    "Density": "1.0e+3 1.0e+3 6.1e+3 1.0e+3 1.0e+3 6.0e+3",
    "BulkModulus": "5.0e+6 5.0e+5 1.0e+4 1.0e+4 1.0e+5 1.0e+5",
    "BulkViscosity": "1.0e-1 5.0e-1 1.0e-1 2.0e+1 1.0e-1 1.0e-3",
    "ShearViscosity": "1.0e-2 5.0e-3 1.0e-3 1.0e-1 1.0e+1 1.0e+0",
    "SurfaceTension": "0.072 0.072 0.00 0.000",
    "YoungModulus": "1.0e8 1e+9 1e+6 1e+6",
    "PoissonRatio": "0.30 0.4 0.20 0.2",
    "Cohesion": "0.0 0.0 0.0 0.0",
    "InternalFrictionAngle": "40.0 20 0.0 0.0",
    "DilatancyFrictionAngle": "0.00 20 20.0 20.0",
    "ActualDebrisSize": "0.001 0.001",
    "InteractionRatio(Type0)": "1.0 1.0 1.0 1.0 1.0 1.0",
    "InteractionRatio(Type1)": "1.0 1.0 1.0 1.0 1.0 1.0",
    "InteractionRatio(Type2)": "1.0 1.0 1.0 1.0 1.0 1.0",
    "InteractionRatio(Type3)": "1.0 1.0 1.0 1.0 1.0 1.0",
    "InteractionRatio(Type4)": "1.0 1.0 1.0 1.0 1.0 1.0",
    "InteractionRatio(Type5)": "1.0 1.0 1.0 1.0 1.0 1.0",
    "Gravity": "0.0 -9.00 0.0",
    "Wall2": "Center 0.0 0.0 0.0    Velocity 0.0  0.0 0.0    Omega 0.0 0.0 0.0",
    "Wall3": "Center 0.0 0.0 0.0    Velocity 0.0  0.0 0.0    Omega 0.0 0.0 0.0",
}

EXACT_DDATA_KEYS = [r[0] for r in EXACT_DDATA_ROWS]
EXACT_DDATA_KIND = {r[0]: r[1] for r in EXACT_DDATA_ROWS}
EXACT_DDATA_COUNT = {r[0]: r[2] for r in EXACT_DDATA_ROWS}
EXACT_DDATA_DESC = {r[0]: r[3] for r in EXACT_DDATA_ROWS}


def _exact_find_line_value(raw: str, key: str) -> str:
    """Find value part after the exact key.  Keeps Wall/Gravity value as raw text."""
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("//") or s.startswith("####"):
            continue
        if s == key:
            return ""
        if s.startswith(key) and (len(s) == len(key) or s[len(key)].isspace()):
            return s[len(key):].strip()
    return EXACT_DDATA_DEFAULTS.get(key, "")


def _normalize_value_count(value: str, key: str) -> str:
    """For numeric array rows, preserve exactly the required row length."""
    kind = EXACT_DDATA_KIND.get(key, "scalar")
    if kind in {"wall", "vector", "scalar"}:
        return str(value).strip()
    vals = str(value).split()
    n = EXACT_DDATA_COUNT.get(key, len(vals))
    defaults = EXACT_DDATA_DEFAULTS.get(key, "").split()
    out = []
    for i in range(n):
        if i < len(vals):
            out.append(vals[i])
        elif i < len(defaults):
            out.append(defaults[i])
        else:
            out.append("0.0")
    return " ".join(out)


def build_exact_ddata_table(raw: str) -> pd.DataFrame:
    rows = []
    for key, kind, count, desc in EXACT_DDATA_ROWS:
        rows.append({
            "Parameter": key,
            "Value": _normalize_value_count(_exact_find_line_value(raw, key), key),
            "Layout": desc,
        })
    return pd.DataFrame(rows)


def render_exact_ddata_from_table(df: pd.DataFrame) -> str:
    values = {str(r.get("Parameter", "")).strip(): str(r.get("Value", "")).strip() for _, r in df.iterrows()}
    lines = ["#######"]
    for key, kind, count, desc in EXACT_DDATA_ROWS:
        val = _normalize_value_count(values.get(key, EXACT_DDATA_DEFAULTS.get(key, "")), key)
        if kind == "wall":
            lines.append(f"{key}    {val}")
        elif kind == "vector":
            lines.append(f"{key}\t\t\t{val}")
        elif kind == "scalar":
            lines.append(f"{key}\t\t\t{val}")
        elif kind == "type0_5":
            v = val.split()
            lines.append(f"{key}\t\t{v[0]}\t  {v[1]}\t{v[2]}\t  {v[3]}\t{v[4]}\t  {v[5]}")
        elif kind == "type0_3":
            v = val.split()
            lines.append(f"{key}\t\t{v[0]}\t{v[1]}\t\t{v[2]}\t\t{v[3]}")
        elif kind == "type2_5":
            v = val.split()
            lines.append(f"{key}\t\t\t{v[0]}\t\t{v[1]}\t\t{v[2]}\t\t{v[3]}")
        elif kind == "type2_3":
            v = val.split()
            lines.append(f"{key}\t\t\t{v[0]}\t\t{v[1]}")
        elif kind == "interaction":
            v = val.split()
            lines.append(f"{key}\t\t\t{v[0]}\t\t{v[1]}\t\t{v[2]}\t\t{v[3]}\t\t{v[4]}\t\t{v[5]}")
        else:
            lines.append(f"{key}\t\t\t{val}")
    return "\n".join(lines) + "\n"

# =============================================================================
# Main tabs
# =============================================================================
tabs = st.tabs([
    "① Generator build",
    "② Source build",
    "③ BOID geometry",
    "④ .data materials",
    "⑤ Particle generation",
    "⑥ Solver & outputs",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Generator make
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    step_bar(1)
    sec(1, "Generator build", "", "c1",
        "Compile the DEM particle generator before anything else.",
        "")

    notice(
        "📂 <b>Generator build path:</b> <code>" + str(generator_build_path) + "</code><br>"
        "Run <code>make</code> here to build the <b>DEM</b> particle generator.<br>"
        "You can continue to the next tab even if the build fails.",
        "blue"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶ make", type="primary", use_container_width=True, key="gen_make"):
            run_with_result(
                st.session_state["generator_build_cmd"],
                generator_build_path, "Generator build log",
                "Generator build completed",
                "Generator build warning/error"
            )
    with c2:
        if st.button("🧹 make clean", use_container_width=True, key="gen_clean"):
            run_with_result(
                st.session_state["generator_clean_cmd"],
                generator_build_path, "Generator clean log",
                "Generator clean completed",
                "Generator clean error"
            )
    with c3:
        if st.button("🔁 clean → make", use_container_width=True, key="gen_rebuild"):
            run_with_result(
                st.session_state["generator_clean_cmd"] + " && " + st.session_state["generator_build_cmd"],
                generator_build_path, "Generator rebuild log",
                "Generator rebuild completed",
                "Generator rebuild error"
            )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    notice(
        "✅ If the build succeeds, <code>generator/DEM</code> will be created.<br>"
        "Next step → <b>② Source build</b> to compile the solver.",
        "green"
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Source make
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    step_bar(2)
    sec(2, "Source build", "", "c2",
        "Choose compiler mode (single core / OpenMP / GPU), configure the Makefile, then compile.",
        "")

    notice(
        "Choose a compiler mode, generate the Makefile, check the content, and save it.<br>"
        "Then build with <b>▶ make source</b>. <b>You can move to the next tab even if an error appears.</b>",
        "blue"
    )

    # Compiler settings
    st.markdown("#### Compiler settings")
    m1, m2 = st.columns([2, 2])
    with m1:
        st.markdown('<div class="help-text">Executable name (TARGET)</div>', unsafe_allow_html=True)
        st.text_input("TARGET", key="mf_target", label_visibility="collapsed")

        st.markdown('<div class="help-text">Compiler mode — choose single-core CPU, OpenMP, or GPU acceleration.</div>', unsafe_allow_html=True)
        st.radio(
            "Compiler mode",
            ["CPU — Single core", "CPU — Multi-core (OpenMP)", "GPU — OpenACC"],
            key="mf_mode", label_visibility="collapsed"
        )
    with m2:
        st.markdown('<div class="help-text">CPU compiler, e.g., g++ or g++-15.</div>', unsafe_allow_html=True)
        st.text_input("CPU compiler (CC_CPU)", key="mf_cc_cpu", label_visibility="collapsed")
        st.markdown('<div class="help-text">GPU compiler, e.g., nvc++; used only for OpenACC mode.</div>', unsafe_allow_html=True)
        st.text_input("GPU compiler (CC_GPU)", key="mf_cc_gpu", label_visibility="collapsed")

    if "GPU" in st.session_state["mf_mode"]:
        st.markdown("#### GPU architecture settings")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown('<div class="help-text">GPU preset — automatically fills the architecture number.</div>', unsafe_allow_html=True)
            preset = st.selectbox("GPU preset", list(GPU_PRESETS.keys()), key="mf_gpu_preset", label_visibility="collapsed")
            if GPU_PRESETS[preset]:
                st.session_state["mf_gpu_cc"] = GPU_PRESETS[preset]
        with g2:
            st.markdown('<div class="help-text">Architecture number, e.g., 89 = Ada Lovelace.</div>', unsafe_allow_html=True)
            st.text_input("GPU cc number", key="mf_gpu_cc", label_visibility="collapsed")

        st.markdown("#### HPCSDK / OpenACC compiler settings")
        notice(
            "Use this when <code>nvc++</code> is installed on the server but is not visible from the GUI shell. "
            "For most servers, use the NVHPC family module directory "
            "<code>/opt/nvidia/hpc_sdk/modulefiles/nvhpc</code>. "
            "Then choose the version separately, e.g. <code>25.11</code>. "
            "If needed, you can still use the root directory "
            "<code>/opt/nvidia/hpc_sdk/modulefiles</code> and a full module name such as <code>nvhpc/25.11</code>.",
            "blue"
        )
        h1, h2 = st.columns(2)
        with h1:
            st.checkbox("Load HPCSDK environment module before source build", key="mf_use_hpcsdk_module")
            st.markdown('<div class="help-text">Default: NVHPC family directory. This is usually the same across servers.</div>', unsafe_allow_html=True)
            st.text_input("HPCSDK module family directory", key="mf_hpcsdk_module_dir", label_visibility="collapsed")
            found_versions = discover_hpcsdk_versions(st.session_state.get("mf_hpcsdk_module_dir", ""))
            version_options = found_versions or ["25.11", "25.9", "25.7", "Custom / manual"]
            current_version = st.session_state.get("mf_hpcsdk_module_version", st.session_state.get("mf_hpcsdk_module_name", "25.11"))
            version_index = version_options.index(current_version) if current_version in version_options else 0
            chosen_version = st.selectbox("HPCSDK version / module name", version_options, index=version_index)
            if chosen_version != "Custom / manual":
                st.session_state["mf_hpcsdk_module_version"] = chosen_version
            st.markdown('<div class="help-text">Optional. Use 25.11 for /modulefiles/nvhpc, or nvhpc/25.11 if using the modulefiles root.</div>', unsafe_allow_html=True)
            st.text_input("Manual HPCSDK version or module name", key="mf_hpcsdk_module_version", label_visibility="collapsed")
            st.session_state["mf_hpcsdk_module_name"] = st.session_state.get("mf_hpcsdk_module_version", "25.11")
        with h2:
            st.checkbox("Use full nvc++ path in generated Makefile", key="mf_use_direct_nvcpp_path")
            st.markdown('<div class="help-text">Optional. If unchecked, Makefile uses CC = nvc++ and the module is loaded only during build.</div>', unsafe_allow_html=True)
            nvcpp_found = discover_nvcpp_paths()
            if nvcpp_found:
                nvcpp_options = nvcpp_found + [st.session_state.get("mf_nvcpp_path", "")]
                nvcpp_options = [x for i, x in enumerate(nvcpp_options) if x and x not in nvcpp_options[:i]]
                current_nvcpp = st.session_state.get("mf_nvcpp_path", nvcpp_options[0])
                nvcpp_index = nvcpp_options.index(current_nvcpp) if current_nvcpp in nvcpp_options else 0
                selected_nvcpp = st.selectbox("Detected nvc++ path", nvcpp_options, index=nvcpp_index)
                st.session_state["mf_nvcpp_path"] = selected_nvcpp
            st.text_input("Manual nvc++ path", key="mf_nvcpp_path", label_visibility="collapsed")

        t1, t2, t3 = st.columns(3)
        with t1:
            if st.button("🔎 Test nvc++", use_container_width=True, key="test_nvcpp"):
                cmd = with_hpcsdk_env(f"{st.session_state.get('mf_nvcpp_path') if st.session_state.get('mf_use_direct_nvcpp_path') else 'nvc++'} --version")
                run_with_result(cmd, source_build_path, "nvc++ test", "nvc++ is available", "nvc++ was not found")
        with t2:
            if st.button("⚙️ Apply module build command", use_container_width=True, key="apply_hpcsdk_make"):
                prefix = hpcsdk_env_prefix()
                st.session_state["source_build_cmd"] = prefix + "make"
                st.session_state["source_clean_cmd"] = prefix + "make clean"
                notice("✅ Source build/clean commands now load the selected HPCSDK module first.", "green")
        with t3:
            if st.button("🎯 Set target to MK-SPH", use_container_width=True, key="set_target_mksph"):
                st.session_state["mf_target"] = "MK-SPH"
                st.session_state["generator_cmd"] = "../../generator/MK-SPH DEM"
                st.session_state["execute_cmd"] = "../../source/MK-SPH dem.data DEM.grid Dam%03d.prof granularImplicit%03d.vtk box.log 4"
                # If scripts were missing or empty, refresh their editor text with the new safe defaults.
                if not st.session_state.get("generate_sh_text", "").strip():
                    st.session_state["generate_sh_text"] = default_generate_script_text()
                if not st.session_state.get("execute_sh_text", "").strip():
                    st.session_state["execute_sh_text"] = default_execute_script_text()
                st.session_state["makefile_editor_text"] = build_makefile_text()
                notice("✅ TARGET and default commands were set to MK-SPH.", "green")

    st.markdown("#### Advanced options")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown('<div class="help-text">Object file list (OBJE)</div>', unsafe_allow_html=True)
        st.text_input("OBJE", key="mf_objects", label_visibility="collapsed")
        st.markdown('<div class="help-text">Linker flags (LDFLAGS), e.g., -lm.</div>', unsafe_allow_html=True)
        st.text_input("LDFLAGS", key="mf_ldflags_base", label_visibility="collapsed")
    with d2:
        st.markdown('<div class="help-text">Additional compiler flags appended to CFLAGS.</div>', unsafe_allow_html=True)
        st.text_input("Additional CFLAGS", key="mf_extra_cflags", label_visibility="collapsed")
        st.markdown('<div class="help-text">Additional linker flags appended to LDFLAGS.</div>', unsafe_allow_html=True)
        st.text_input("Additional LDFLAGS", key="mf_extra_ldflags", label_visibility="collapsed")
    with d3:
        st.markdown('<div class="help-text">WFLAGS, e.g., warning flags.</div>', unsafe_allow_html=True)
        st.text_input("WFLAGS", key="mf_wflags", label_visibility="collapsed")
        st.markdown('<div class="help-text">clean rule: files removed by make clean.</div>', unsafe_allow_html=True)
        st.text_input("clean rule", key="mf_clean_rule", label_visibility="collapsed")

    cb1, cb2 = st.columns(2)
    with cb1:
        st.checkbox("Add debug flag (-g)", key="mf_use_debug",
                    help="-g helps debugging with gdb, but it can reduce performance.")
    with cb2:
        st.checkbox("Add warning flag (-Wall)", key="mf_use_wall",
                    help="Show detailed compiler warnings.")

    st.checkbox("Back up the existing Makefile before saving (.bak_timestamp)", key="mf_backup_before_save")

    st.markdown("#### Makefile editor")
    notice("Generate the Makefile from the settings below, or load and edit an existing Makefile.", "blue")

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("📋 Generate Makefile from settings", use_container_width=True, key="mf_gen"):
            st.session_state["makefile_editor_text"] = build_makefile_text()
    with b2:
        if st.button("📂 Load existing Makefile", use_container_width=True, key="mf_load"):
            st.session_state["makefile_editor_text"] = read_text(source_makefile_path, build_makefile_text())
    with b3:
        if st.button("💾 Save Makefile", type="primary", use_container_width=True, key="mf_save"):
            if st.session_state["mf_backup_before_save"] and source_makefile_path.exists():
                bak = backup_file(source_makefile_path)
                notice(f"📦 Backup created: <code>{bak}</code>", "blue")
            write_text(source_makefile_path, st.session_state["makefile_editor_text"])
            notice(f"✅ Saved: <code>{source_makefile_path}</code>", "green")

    if not st.session_state.get("makefile_editor_text"):
        st.session_state["makefile_editor_text"] = build_makefile_text()

    st.text_area("Makefile", key="makefile_editor_text", height=280,
                 help="You can edit this directly. Press Save to write it to source/makefile.")

    st.markdown(f'<div class="help-text">Source build path: <code>{source_build_path}</code></div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        if st.button("▶ make source", type="primary", use_container_width=True, key="src_make"):
            run_with_result(
                with_hpcsdk_env(st.session_state["source_build_cmd"]), source_build_path,
                "Source build log", "Source build completed", "Source build warning/error"
            )
    with r2:
        if st.button("🧹 make clean", use_container_width=True, key="src_clean"):
            run_with_result(
                with_hpcsdk_env(st.session_state["source_clean_cmd"]), source_build_path,
                "Source clean log", "Source clean completed", "Source clean error"
            )
    with r3:
        if st.button("🔁 clean → make", use_container_width=True, key="src_rebuild"):
            run_with_result(
                with_hpcsdk_env(st.session_state["source_clean_cmd"] + " && " + st.session_state["source_build_cmd"]),
                source_build_path, "Source rebuild log", "Source rebuild completed", "Source rebuild warning/error"
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — BOID
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    step_bar(3)
    sec(3, "BOID geometry editor", "", "c3",
        "Define particle domain and shapes for the particle generator.",
        "")

    st.markdown(f'<div class="help-text">Current case folder: <code>{workdir}</code></div>', unsafe_allow_html=True)
    st.dataframe(list_result_files(workdir), use_container_width=True, hide_index=True)

    st.markdown("#### Basic settings")
    notice(
        "<b>2D mode</b>: the Z direction is fixed to <code>ZMin=0.0</code> and <code>ZMax=ParticleDistance</code>.<br>"
        "<b>3D mode</b>: enter the minimum and maximum Z coordinates.",
        "blue"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="help-text">Dimension: in 2D, ZMin=0.0 and ZMax=ParticleDistance are fixed.</div>', unsafe_allow_html=True)
        st.radio("Dimension", ["2D", "3D"], key="boid_dim", horizontal=True, label_visibility="collapsed")
        st.markdown('<div class="help-text">Particle spacing (ParticleDistance). Smaller spacing increases computational cost.</div>', unsafe_allow_html=True)
        st.number_input("Particle spacing", min_value=1e-12, format="%.8g", key="boid_particle_distance",
                        label_visibility="collapsed")
    with c2:
        st.markdown('<div class="help-text">Computational domain in X (LowerDomain / UpperDomain).</div>', unsafe_allow_html=True)
        st.number_input("X min", format="%.8g", key="boid_lower_x")
        st.number_input("X max", format="%.8g", key="boid_upper_x")
    with c3:
        st.markdown('<div class="help-text">Computational domain in Y.</div>', unsafe_allow_html=True)
        st.number_input("Y min", format="%.8g", key="boid_lower_y")
        st.number_input("Y max", format="%.8g", key="boid_upper_y")

    if st.session_state["boid_dim"] == "3D":
        z1, z2 = st.columns(2)
        with z1:
            st.markdown('<div class="help-text">Computational domain in Z (3D only).</div>', unsafe_allow_html=True)
            st.number_input("Z min", format="%.8g", key="boid_lower_z")
        with z2:
            st.markdown('<br>', unsafe_allow_html=True)
            st.number_input("Z max", format="%.8g", key="boid_upper_z")

    st.markdown("#### Shape editor")
    notice(
         "<b>Kind</b> — box: cuboid/rectangle, circle/sphere, cylinder, triangle<br>"
        "<b>Mode</b> — add: add particles, remove: subtract particles<br>"
        "<b>Type</b> — integer particle type. Example: 0=Fluid 0, 1=Fluid 1, 2=Structure 1, 3=Structure 2, 4=Wall, 5=Moving rigid body<br>"
        "<b>RigidType</b> — rigid body group ID. -1 = not rigid<br>"
        "<b>box</b>: uses XMin/XMax/YMin/YMax. "
        "<b>circle/cylinder</b>: uses CenterX/Y/Z and Radius.",
        "blue"
    )

    type_rows = [{"Type": k, "Description": v[0]} for k, v in PARTICLE_TYPE_HELP.items()]
    with st.expander("📘 Particle type guide", expanded=True):
        st.dataframe(pd.DataFrame(type_rows), use_container_width=True, hide_index=True)
        st.markdown("""
<div class="param-help">
<b>RigidType</b>: -1 = not rigid; 0 or larger = same rigid-body group ID.<br>
<b>Mode = remove</b>: subtracts particles from shapes that were added before it.
</div>
""", unsafe_allow_html=True)

    # Type and RigidType must stay int — use format="%d" and int64 dtype
    edited_df = st.data_editor(
        st.session_state["boid_shapes"],
        key="boid_shape_editor",
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Name":      st.column_config.TextColumn("Name", help="Arbitrary shape name."),
            "Kind":      st.column_config.SelectboxColumn("Kind", options=["box", "circle", "cylinder", "sphere", "triangle"]),
            "Mode":      st.column_config.SelectboxColumn("Mode", options=["add", "remove"]),
            "Type":      st.column_config.NumberColumn("Type\nparticle type", step=1, format="%d", help="Integer. Example: 0=Fluid 0, 1=Fluid 1, 2=Structure 1, 3=Structure 2, 4=Wall, 5=Moving rigid body."),
            "RigidType": st.column_config.NumberColumn("RigidType\nrigid ID", step=1, format="%d", help="Integer. -1 = not rigid; 0 or larger = rigid-body group ID."),
            "XMin": st.column_config.NumberColumn("XMin", format="%.6g"),
            "XMax": st.column_config.NumberColumn("XMax", format="%.6g"),
            "YMin": st.column_config.NumberColumn("YMin", format="%.6g"),
            "YMax": st.column_config.NumberColumn("YMax", format="%.6g"),
            "ZMin": st.column_config.NumberColumn("ZMin", format="%.6g"),
            "ZMax": st.column_config.NumberColumn("ZMax", format="%.6g"),
            "CenterX": st.column_config.NumberColumn("CenterX", format="%.6g"),
            "CenterY": st.column_config.NumberColumn("CenterY", format="%.6g"),
            "CenterZ": st.column_config.NumberColumn("CenterZ", format="%.6g"),
            "Radius":   st.column_config.NumberColumn("Radius", format="%.6g"),
            "Height":   st.column_config.NumberColumn("Height", format="%.6g"),
            "AngleDeg": st.column_config.NumberColumn("AngleDeg", format="%.4g"),
            "VelocityX": st.column_config.NumberColumn("Vx", format="%.6g"),
            "VelocityY": st.column_config.NumberColumn("Vy", format="%.6g"),
            "VelocityZ": st.column_config.NumberColumn("Vz", format="%.6g"),
            "Enthalpy":  st.column_config.NumberColumn("Enthalpy", format="%.6g"),
        },
    )
    # Keep Type / RigidType as int
    if edited_df is not None:
        edited_df["Type"] = edited_df["Type"].fillna(2).astype("int64")
        edited_df["RigidType"] = edited_df["RigidType"].fillna(-1).astype("int64")
        st.session_state["boid_shapes"] = edited_df

    boid_warnings = validate_boid_shapes(st.session_state["boid_shapes"], st.session_state.get("boid_dim", "2D"))
    show_validation_box("BOID shape validation", boid_warnings)

    # Shape wizard
    with st.expander("➕ Add shape wizard", expanded=False):
        notice("Enter a new shape and press Add.", "blue")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            new_name  = st.text_input("Name", value="new_shape", key="wiz_name")
            new_kind  = st.selectbox("Kind", ["box", "circle", "cylinder", "sphere", "triangle"], key="wiz_kind")
            new_mode  = st.selectbox("Mode", ["add", "remove"], key="wiz_mode")
        with a2:
            new_type  = st.number_input("Type", value=2, step=1, key="wiz_type",
                                         help="0=Fluid 0, 1=Fluid 1, 2=Structure 1, 3=Structure 2, 4=Wall, 5=Moving rigid body.")
            new_rigid = st.number_input("RigidType", value=-1, step=1, key="wiz_rigid",
                                         help="-1 = not rigid; 0 or larger = rigid-body group ID.")
            new_angle = st.number_input("AngleDeg", value=0.0, key="wiz_angle")
        with a3:
            new_x0 = st.number_input("X min or center X", value=0.0, key="wiz_x0")
            new_x1 = st.number_input("X max (box only)", value=0.1, key="wiz_x1")
            new_y0 = st.number_input("Y min or center Y", value=0.0, key="wiz_y0")
            new_y1 = st.number_input("Y max (box only)", value=0.1, key="wiz_y1")
        with a4:
            new_radius   = st.number_input("Radius (circle/cylinder/sphere)", value=0.01, key="wiz_radius")
            new_height   = st.number_input("Height (cylinder/triangle, 3D)", value=0.01, key="wiz_height")
            new_enthalpy = st.number_input("Enthalpy", value=0.0, key="wiz_enthalpy")

        if st.button("➕ Add this shape", use_container_width=True, key="wiz_add"):
            pdist = float(st.session_state["boid_particle_distance"])
            if new_kind == "box":
                row = [new_name, new_kind, new_mode, int(new_type), int(new_rigid),
                       new_x0, new_x1, new_y0, new_y1, 0.0, pdist,
                       0.5*(new_x0+new_x1), 0.5*(new_y0+new_y1), 0.5*pdist,
                       0.0, 0.0, new_angle, 0.0, 0.0, 0.0, new_enthalpy]
            else:
                row = [new_name, new_kind, new_mode, int(new_type), int(new_rigid),
                       new_x0-new_radius, new_x0+new_radius, new_y0-new_radius, new_y0+new_radius, 0.0, pdist,
                       new_x0, new_y0, 0.5*pdist, new_radius, new_height, new_angle, 0.0, 0.0, 0.0, new_enthalpy]
            new_row_df = pd.DataFrame([row], columns=BOID_COLUMNS)
            new_row_df["Type"] = new_row_df["Type"].astype("int64")
            new_row_df["RigidType"] = new_row_df["RigidType"].astype("int64")
            st.session_state["boid_shapes"] = pd.concat(
                [st.session_state["boid_shapes"], new_row_df], ignore_index=True
            )
            st.rerun()

    # BOID actions
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("📂 Load .boid", use_container_width=True, key="boid_load"):
            text = read_text(boid_file, render_boid_text())
            st.session_state["raw_boid_text"] = text
            st.session_state["boid_editor_text"] = text
            parse_boid_text(text)
            notice(f"✅ Loaded: <code>{boid_file}</code>", "green")
    with b2:
        if st.button("🔄 Apply form to raw text", use_container_width=True, key="boid_reflect"):
            text = render_boid_text()
            st.session_state["raw_boid_text"] = text
            st.session_state["boid_editor_text"] = text
            notice("The form content was applied to the raw text below.", "blue")
    with b3:
        if st.button("💾 Save .boid", type="primary", use_container_width=True, key="boid_save"):
            # Save the raw editor text. Press "Apply form to raw text" first
            # if you want to save the current table/wizard form values.
            text = st.session_state.get("boid_editor_text", render_boid_text())
            st.session_state["raw_boid_text"] = text
            write_text(boid_file, text)
            st.session_state["boid_loaded_path"] = str(boid_file)
            notice(f"✅ Saved: <code>{boid_file}</code>", "green")

    st.markdown("##### Raw .boid file")
    st.text_area(".boid raw", key="boid_editor_text", height=320, label_visibility="collapsed",
                 help="You can edit the raw .boid text directly. Press Save .boid to write it to disk.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — .data editor
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    step_bar(4)
    sec(4, ".data editor", "", "c4",
        "Edit the exact MK-SPH .data format without changing row order, row length, or key names.",
        "")

    notice(
        f"Target file: <code>{case_file}</code><br>"
        "This editor follows your exact .data format. It does not convert rows into another style. "
        "Every key is saved in the same order: Dt, ElasticDt, material rows, InteractionRatio, Gravity, Wall2, Wall3.",
        "blue"
    )

    if "exact_ddata_table" not in st.session_state:
        st.session_state["exact_ddata_table"] = build_exact_ddata_table(st.session_state.get("raw_case_text", ""))

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        if st.button("📂 Load .data", use_container_width=True, key="data_load_exact"):
            st.session_state["raw_case_text"] = read_text(case_file, "")
            st.session_state["exact_ddata_table"] = build_exact_ddata_table(st.session_state["raw_case_text"])
            st.session_state["case_loaded_path"] = str(case_file)
            notice(f"✅ Loaded: <code>{case_file}</code>", "green")
            st.rerun()
    with c2:
        if st.button("⬇️ Apply table", use_container_width=True, key="data_apply_exact"):
            st.session_state["raw_case_text"] = render_exact_ddata_from_table(st.session_state["exact_ddata_table"])
            notice("The table was applied to raw .data text using the exact MK-SPH format.", "green")
    with c3:
        if st.button("💾 Save .data", type="primary", use_container_width=True, key="data_save_exact"):
            text_to_save = st.session_state.get("raw_case_text", "")
            if not text_to_save.strip():
                text_to_save = render_exact_ddata_from_table(st.session_state.get("exact_ddata_table", build_exact_ddata_table("")))
                st.session_state["raw_case_text"] = text_to_save
            write_text(case_file, text_to_save)
            st.session_state["case_loaded_path"] = str(case_file)
            notice(f"✅ Saved without changing the .data format: <code>{case_file}</code>", "green")
    with c4:
        st.markdown(
            f'<div class="help-text" style="padding-top:8px;">File: <code>{case_file}</code> — '
            f'{"✅ exists" if case_file.exists() else "⚠️ not created yet"}</div>',
            unsafe_allow_html=True
        )

    data_tabs = st.tabs(["Exact .data table", "Raw .data text", "Format reference"])

    with data_tabs[0]:
        st.markdown("#### Exact MK-SPH `.data` parameters")
        notice(
            "Edit the <b>Value</b> column only. For array rows, keep values separated by spaces. "
            "The GUI enforces the required number of values: 6 for Type0–5 rows, 4 for Type2–5 rows, "
            "2 for ActualDebrisSize, and 6 for each InteractionRatio row.",
            "yellow"
        )
        edited_exact_df = st.data_editor(
            st.session_state["exact_ddata_table"],
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            height=720,
            column_config={
                "Parameter": st.column_config.TextColumn("Parameter", disabled=True),
                "Value": st.column_config.TextColumn("Value"),
                "Layout": st.column_config.TextColumn("Layout", disabled=True),
            },
            key="exact_ddata_editor",
        )
        st.session_state["exact_ddata_table"] = pd.DataFrame(edited_exact_df)

        e1, e2, e3 = st.columns(3)
        with e1:
            if st.button("🔄 Re-parse from raw text", use_container_width=True, key="exact_reparse_from_raw"):
                st.session_state["exact_ddata_table"] = build_exact_ddata_table(st.session_state.get("raw_case_text", ""))
                notice("Re-parsed the exact table from the raw .data text.", "blue")
                st.rerun()
        with e2:
            if st.button("⬇️ Apply exact table to raw text", use_container_width=True, key="exact_apply_to_raw"):
                st.session_state["raw_case_text"] = render_exact_ddata_from_table(st.session_state["exact_ddata_table"])
                notice("Applied exact table to raw .data text.", "green")
        with e3:
            if st.button("💾 Apply exact table and save", type="primary", use_container_width=True, key="exact_apply_save"):
                st.session_state["raw_case_text"] = render_exact_ddata_from_table(st.session_state["exact_ddata_table"])
                write_text(case_file, st.session_state["raw_case_text"])
                st.session_state["case_loaded_path"] = str(case_file)
                notice(f"✅ Applied exact format and saved: <code>{case_file}</code>", "green")

    with data_tabs[1]:
        st.markdown("#### Raw `.data` text")
        notice(
            "Direct text edit is allowed. Press <b>Re-parse from raw text</b> in the table tab after changing raw text. "
            "Saving raw text writes exactly what is shown here.",
            "blue"
        )
        st.text_area(
            "Raw .data file",
            key="raw_case_text",
            height=760,
            label_visibility="collapsed",
            help="Directly edit the exact .data file text."
        )

    with data_tabs[2]:
        st.markdown("#### Required format")
        st.code("""#######
Dt			1.0e-4
ElasticDt			1.0e-4
OutputInterval			1.0
VtkOutputInterval			1.0e-2
EndTime			2.0e1
RadiusRatioA			2.5
RadiusRatioP			2.5
RadiusRatioV			2.5
Density		1.0e+3	  1.0e+3	6.1e+3	  1.0e+3	1.0e+3	  6.0e+3
BulkModulus	5.0e+6	  5.0e+5	 1.0e+4	  1.0e+4	 1.0e+5	  1.0e+5
BulkViscosity	1.0e-1	  5.0e-1	 1.0e-1	  2.0e+1	1.0e-1	  1.0e-3
ShearViscosity	1.0e-2	  5.0e-3	1.0e-3	  1.0e-1	1.0e+1	  1.0e+0
SurfaceTension		0.072	0.072		0.00		0.000
YoungModulus			1.0e8		1e+9		1e+6		1e+6
PoissonRatio			0.30		0.4		0.20		0.2
Cohesion			0.0		0.0		0.0		0.0
InternalFrictionAngle			40.0		20		0.0		0.0
DilatancyFrictionAngle			0.00		20		20.0		20.0
ActualDebrisSize			0.001		0.001	
InteractionRatio(Type0)			1.0		1.0		1.0		1.0		1.0		1.0
InteractionRatio(Type1)			1.0		1.0		1.0		1.0		1.0		1.0
InteractionRatio(Type2)			1.0		1.0		1.0		1.0		1.0		1.0
InteractionRatio(Type3)			1.0		1.0		1.0		1.0		1.0		1.0
InteractionRatio(Type4)			1.0		1.0		1.0		1.0		1.0		1.0
InteractionRatio(Type5)			1.0		1.0		1.0		1.0		1.0		1.0
Gravity			0.0 -9.00 0.0
Wall2    Center 0.0 0.0 0.0    Velocity 0.0  0.0 0.0    Omega 0.0 0.0 0.0
Wall3    Center 0.0 0.0 0.0    Velocity 0.0  0.0 0.0    Omega 0.0 0.0 0.0""", language="text")

# TAB 5 — Generate particles
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    step_bar(5)
    sec(5, "Particle generation (./generate.sh)", "", "c5",
        "Run the particle generator to create .grid/.prof files from .boid and .data.",
        "")

    notice(
        f"Working directory: <code>{workdir}</code><br>"
        f".boid: <code>{boid_file}</code><br>"
        f".data: <code>{case_file}</code><br>"
        "The Run button automatically saves .boid and .data before running generate.sh.",
        "blue"
    )

    st.markdown("##### generate.sh content")
    st.text_area("generate.sh", key="generate_sh_text", height=220, label_visibility="collapsed",
                 help="Write the particle generation command, e.g., ../../generator/MK-SPH DEM. Save or Run applies it.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("💾 Save generate.sh", use_container_width=True, key="gsh_save"):
            save_generate_sh(generate_sh)
            notice("✅ generate.sh saved", "green")
    with c2:
        if st.button("💾 Save .boid/.data", use_container_width=True, key="gsh_save_inputs"):
            # Tab 3 already instantiated the BOID text_area widget in this run.
            # Therefore, do not assign to a widget-backed key here.
            latest_boid_text = render_boid_text()
            write_text(case_file, st.session_state.get("raw_case_text", ""))
            write_text(boid_file, latest_boid_text)
            notice("✅ .boid and .data saved", "green")
    with c3:
        if st.button("▶ Generate particles", type="primary", use_container_width=True, key="gsh_run_bg"):
            # Directly write the latest form-rendered BOID text.
            # Do not modify boid_editor_text/raw_boid_text here after the text_area has been instantiated.
            latest_boid_text = render_boid_text()
            write_text(case_file, st.session_state.get("raw_case_text", ""))
            write_text(boid_file, latest_boid_text)
            save_generate_sh(generate_sh)
            start_background_cmd("./generate.sh", workdir, workdir / "gui_generate.log", "generate_pid", "generate.sh")
    with c4:
        if st.button("⏹ Stop generation", use_container_width=True, key="gsh_stop"):
            stop_background_cmd("generate_pid", "generate.sh")

    show_process_status("generate_pid", "generate.sh")

    st.markdown("#### Shape preview after generation")
    p1, p2 = st.columns([1, 3])
    with p1:
        if st.button("🔍 Preview generated particles", use_container_width=True, key="preview_generated"):
            st.session_state["show_preview"] = True
    with p2:
        st.markdown('<div class="help-text">2D uses an XY scatter plot. 3D uses a 3D scatter plot. The newest .vtk/.prof/.grid file is detected automatically.</div>', unsafe_allow_html=True)
    if st.session_state.get("show_preview", False):
        show_particle_preview(workdir, st.session_state.get("boid_dim", "2D"))

    notice(
        "⚠️ generate.sh runs in the background. You can watch the log, stop the process, and preview particles.<br>"
        "It runs in the background, so the GUI remains usable.",
        "yellow"
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — Execute and results
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    step_bar(6)
    sec(6, "Solver execution & outputs", "", "c6",
        "Run the main MPH solver and inspect output files (.vtk, .prof, .log, etc.).",
        "")

    notice(
        f"Working directory: <code>{workdir}</code><br>"
        "The Run button saves .data before running execute.sh.",
        "blue"
    )

    st.markdown("##### execute.sh content")
    st.text_area("execute.sh", key="execute_sh_text", height=220, label_visibility="collapsed",
                 help="Write the solver command, e.g., ../../source/Mph_Elastic_Explicit dem.data ...")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("💾 Save execute.sh", use_container_width=True, key="esh_save"):
            save_execute_sh(execute_sh)
            notice("✅ execute.sh saved", "green")
    with c2:
        if st.button("▶ Run solver", type="primary", use_container_width=True, key="esh_run_bg"):
            write_text(case_file, st.session_state.get("raw_case_text", ""))
            save_execute_sh(execute_sh)
            start_background_cmd("./execute.sh", workdir, workdir / "gui_execute.log", "execute_pid", "execute.sh")
    with c3:
        if st.button("⏹ Stop solver", use_container_width=True, key="esh_stop"):
            stop_background_cmd("execute_pid", "execute.sh")
    with c4:
        if st.button("🔄 Refresh outputs", use_container_width=True, key="esh_refresh"):
            st.rerun()

    show_process_status("execute_pid", "execute.sh")

    notice(
        "⚠️ execute.sh runs in the background. The Stop button terminates the whole process group using SIGTERM and then SIGKILL if needed.<br>"
        "The solver runs in the background. Stop sends SIGTERM and then SIGKILL if needed.",
        "yellow"
    )

    st.markdown("#### Recent outputs")
    recent_files = list_recent_output_files(results_dir, limit=12)
    if recent_files:
        selected_output = st.selectbox("Inspect/download recent output", [p.name for p in recent_files])
        selected_path = next(p for p in recent_files if p.name == selected_output)
        st.markdown(
            f'<div class="help-text">Selected: <code>{selected_path}</code> '
            f'({selected_path.stat().st_size:,} bytes)</div>',
            unsafe_allow_html=True,
        )
        try:
            preview_text = selected_path.read_text(encoding="utf-8", errors="replace")[:8000]
            st.code(preview_text if preview_text else "(empty file)", language="text")
        except Exception:
            notice("This file is not text-previewable, but it can still be downloaded.", "yellow")
        st.download_button(
            "⬇️ Download selected output",
            data=selected_path.read_bytes(),
            file_name=selected_path.name,
            mime="application/octet-stream",
            use_container_width=True,
        )
    else:
        notice("No recent output files were found yet.", "yellow")

    st.markdown("#### Output files")
    st.markdown(f'<div class="help-text">Directory: <code>{results_dir}</code></div>', unsafe_allow_html=True)
    st.dataframe(list_result_files(results_dir), use_container_width=True, hide_index=True)

# =============================================================================
# Footer
# =============================================================================
with st.expander("📖 Launch instructions"):
    st.markdown(f"""
```bash
cd {project_root}
python -m streamlit run GUI/streamlit_app.py
```

**Workflow:**
1. **Generator build** — compile the DEM particle generator with `make`.
2. **Source build** — choose compiler mode and compile the solver with `make`.
3. **BOID geometry** — set the domain and particle shapes in `.boid`.
4. **.data materials** — edit global solver parameters and material properties.
5. **Particle generation** — run `./generate.sh` to create particle placement files.
6. **Solver & outputs** — run `./execute.sh`, stop the solver if needed, and check the output files.

**Main features in this version:**
- English-only GUI labels.
- Separate material editing for Fluid 0, Fluid 1, Structure 1, Structure 2, Wall, and Moving rigid body type 5.
- Stop buttons for background `generate.sh` and `execute.sh`.
- 2D/3D particle preview after generation.
- Raw `.data` and `.boid` editors remain available for advanced settings.
- BOID validation warnings before generation.
- Recent output preview and download panel.
- Automatic reload when switching case/data/boid paths.
""")

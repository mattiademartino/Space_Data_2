"""
plot.py — Temperature vs time across the 3 CanSat datasets.

Produces two figures saved in Report/Img/:
  fig_temp_overlay.png   — GRASP, VAMOS, OBAMA overlaid (common time axis, minutes)
  fig_temp_separate.png  — 3 individual subplots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from utils import FIG_DIR, save_figure

# ── Constants ────────────────────────────────────────────────────────────────
DATA     = Path("CanSat data utili")
UWYO_CSV = Path("dati/uwyo_06610_2026-02-05_12Z.csv")

P0, T0_K, L, g, Rd = 101325.0, 288.15, 6.5e-3, 9.80665, 287.058

def isa_alt(P_pa):
    return (T0_K / L) * (1.0 - (np.asarray(P_pa) / P0) ** (Rd * L / g))

# ── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":         120,
    "font.size":          10,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

# ── Load datasets ─────────────────────────────────────────────────────────────
def load_grasp():
    df = pd.read_csv(
        DATA / "science_GRASP.csv",
        usecols=[0, 2, 3, 4, 5, 6],
        names=["t_ms", "temp_C", "press_Pa", "alt_m", "pm25", "pm10"],
        header=0, skipfooter=8, engine="python",
    )
    df = df.dropna().astype(float)
    df["t_min"] = (df["t_ms"] - df["t_ms"].iat[0]) / 60000   # ms → min
    return df.iloc[1:].copy()   # drop sensor-init row 0


def load_vamos():
    df = pd.read_csv(DATA / "science_VAMOS.csv")
    df.columns = ["t_ms", "co2_ppm", "temp_C", "press_hPa"]
    df = (df[df["t_ms"] != "timestamp_ms"]
            .apply(pd.to_numeric, errors="coerce").dropna())
    df["t_min"] = (df["t_ms"] - df["t_ms"].iat[0]) / 60000
    return df.reset_index(drop=True)


def load_obama():
    df = pd.read_excel(DATA / "OBAMA_data_decoded.xlsx", sheet_name="decoded")
    df = df.apply(pd.to_numeric, errors="coerce")
    if "Time_s" in df.columns:
        df["t_min"] = (df["Time_s"] - df["Time_s"].iat[0]) / 60
    else:
        df["t_min"] = np.arange(len(df)) / 60
    return df


grasp = load_grasp()
vamos = load_vamos()
obama = load_obama()

# ── OBAMA: filter valid temperature rows ──────────────────────────────────────
mask1 = obama["first_temp_avg_C"].between(-20, 50)
mask2 = obama["second_temp_avg_C"].between(-20, 50)

# ── Palette ───────────────────────────────────────────────────────────────────
C_GRASP = "tab:red"
C_VAMOS = "tab:blue"
C_OBAMA = "tab:green"

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — overlaid (temperature vs time in minutes)
# UWYO is a sounding (no time axis) → shown as a small inset vs altitude
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.suptitle("Temperatura — 4 dataset sovrapposti", fontweight="bold")

ax.plot(
    grasp["t_min"], grasp["temp_C"],
    lw=0.9, color=C_GRASP, alpha=0.85, label="GRASP",
)
ax.plot(
    vamos["t_min"], vamos["temp_C"],
    lw=0.6, color=C_VAMOS, alpha=0.7, label="VAMOS-science",
)
if mask1.any():
    ax.plot(
        obama.loc[mask1, "t_min"], obama.loc[mask1, "first_temp_avg_C"],
        "o-", ms=4, lw=1.2, color=C_OBAMA, label="OBAMA sensor 1",
    )
if mask2.any():
    ax.plot(
        obama.loc[mask2, "t_min"], obama.loc[mask2, "second_temp_avg_C"],
        "D--", ms=4, lw=1.0, color=C_OBAMA, alpha=0.55, label="OBAMA sensor 2",
    )

ax.set_xlabel("Tempo relativo (min)")
ax.set_ylabel("Temperatura (°C)")
ax.legend(fontsize=9)

fig.tight_layout()
save_figure(fig, "fig_temp_overlay.png")
print("Saved → Report/Img/fig_temp_overlay.png")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — 3 subplot separati (1×3)
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle("Temperatura — subplot individuali", fontweight="bold", fontsize=13)

# ── GRASP ─────────────────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(grasp["t_min"], grasp["temp_C"], lw=0.9, color=C_GRASP, alpha=0.85)
ax.set_xlabel("Tempo da espulsione (min)")
ax.set_ylabel("Temperatura (°C)")
ax.set_title("GRASP")

# ── VAMOS-science ─────────────────────────────────────────────────────────────
ax = axes[1]
ax.plot(vamos["t_min"], vamos["temp_C"], lw=0.5, color=C_VAMOS, alpha=0.8)
ax.set_xlabel("Tempo da inizio (min)")
ax.set_ylabel("Temperatura (°C)")
ax.set_title("VAMOS-science")

# ── OBAMA ─────────────────────────────────────────────────────────────────────
ax = axes[2]
if mask1.any():
    ax.plot(
        obama.loc[mask1, "t_min"], obama.loc[mask1, "first_temp_avg_C"],
        "o-", ms=4, lw=1.2, color=C_OBAMA, label="Sensor 1",
    )
if mask2.any():
    ax.plot(
        obama.loc[mask2, "t_min"], obama.loc[mask2, "second_temp_avg_C"],
        "D--", ms=4, lw=1.0, color=C_OBAMA, alpha=0.55, label="Sensor 2",
    )
ax.set_xlabel("Tempo relativo (min)")
ax.set_ylabel("Temperatura (°C)")
ax.set_title("OBAMA")
ax.legend(fontsize=9)

fig2.tight_layout()
save_figure(fig2, "fig_temp_separate.png")
print("Saved → Report/Img/fig_temp_separate.png")
plt.show()

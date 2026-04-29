
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────
# 0.  PATHS  — adjust if needed
# ─────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH     = os.path.join(SCRIPT_DIR, "uwyo_06610_2026-02-05_12Z.csv")
OUTPUT_DIR   = SCRIPT_DIR          # saves next to the CSV (alice/Payerne/)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
 
# Strip whitespace from column names
df.columns = df.columns.str.strip()
 
# Select and rename for convenience
df = df.rename(columns={
    "pressure_hPa"               : "pres",
    "geopotential height_m"      : "hght",
    "temperature_C"              : "temp",
    "dew point temperature_C"    : "dwpt",
    "relative humidity_%"        : "relh",
    "wind direction_degree"      : "wdir",
    "wind speed_m/s"             : "wspd",
})
 
# Convert to numeric, drop bad rows
for col in ["pres", "hght", "temp", "dwpt", "relh", "wdir", "wspd"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
 
df = df.dropna(subset=["pres", "hght", "temp", "dwpt"]).reset_index(drop=True)
 
# Sort by increasing altitude / decreasing pressure
df = df.sort_values("hght").reset_index(drop=True)
 
# ── FILTER: keep only levels from surface down to 800 hPa ──
df = df[df["pres"] >= 800].reset_index(drop=True)
 
print(f"Loaded {len(df)} levels  |  "
      f"Alt range: {df['hght'].min():.0f} – {df['hght'].max():.0f} m  |  "
      f"Pres range: {df['pres'].max():.1f} – {df['pres'].min():.1f} hPa")
 
# ─────────────────────────────────────────────
# STYLE SETTINGS
# ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family"    : "DejaVu Sans",
    "font.size"      : 11,
    "axes.linewidth" : 1.2,
    "axes.grid"      : True,
    "grid.alpha"     : 0.35,
    "grid.linestyle" : "--",
    "lines.linewidth": 1.8,
})
 
COLOR_TEMP = "#d62728"    # red
COLOR_DWPT = "#1f77b4"    # blue
COLOR_RH   = "#2ca02c"    # green
STATION    = "Payerne (06610)"
DATE_STR   = "2026-02-05  12 UTC"
 
 
# ═══════════════════════════════════════════════
# FIGURE 1 — Skew-T style: T & Td vs PRESSURE
# ═══════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(7, 9))
 
ax1.plot(df["temp"], df["pres"],  color=COLOR_TEMP, label="Temperature (°C)")
ax1.plot(df["dwpt"], df["pres"],  color=COLOR_DWPT, label="Dew Point (°C)", linestyle="--")
 
# Pressure axis: log scale, inverted (surface at bottom)
ax1.set_yscale("log")
ax1.invert_yaxis()
ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax1.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1, 2, 3, 4, 5, 7]))
ax1.yaxis.set_minor_formatter(ticker.NullFormatter())
 
# Standard pressure levels
std_levels = [1000, 925, 850, 800]
pmin, pmax = df["pres"].min(), df["pres"].max()
std_levels = [p for p in std_levels if pmin <= p <= pmax]
ax1.set_yticks(std_levels)
ax1.set_yticklabels([str(p) for p in std_levels])
 
# Freezing line
ax1.axvline(0, color="gray", linewidth=1.0, linestyle=":", alpha=0.7, label="0 °C")
 
ax1.set_xlabel("Temperature / Dew Point (°C)", fontsize=12)
ax1.set_ylabel("Pressure (hPa)", fontsize=12)
ax1.set_title(f"Vertical Profile — {STATION}\n{DATE_STR}", fontsize=13, fontweight="bold")
ax1.legend(loc="upper right", framealpha=0.9)
 
fig1.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "profile_T_Td_vs_pressure.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved → {out1}")
plt.close(fig1)
 
 
# ═══════════════════════════════════════════════
# FIGURE 2 — T & Td vs GEOPOTENTIAL HEIGHT
# ═══════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(7, 9))
 
ax2.plot(df["temp"], df["hght"] / 1000, color=COLOR_TEMP, label="Temperature (°C)")
ax2.plot(df["dwpt"], df["hght"] / 1000, color=COLOR_DWPT, label="Dew Point (°C)", linestyle="--")
ax2.axvline(0, color="gray", linewidth=1.0, linestyle=":", alpha=0.7, label="0 °C")
 
# Secondary y-axis with pressure
ax2b = ax2.twinx()
# Map pressure to height for tick positions
from scipy.interpolate import interp1d
p2h = interp1d(df["hght"] / 1000, df["pres"], bounds_error=False, fill_value="extrapolate")
h2p = interp1d(df["pres"], df["hght"] / 1000, bounds_error=False, fill_value="extrapolate")
 
hmax = df["hght"].max() / 1000
hmin = df["hght"].min() / 1000
std_levels_filtered = [p for p in [1000, 925, 850, 800]
                       if pmin <= p <= pmax]
tick_heights = [float(h2p(p)) for p in std_levels_filtered]
ax2b.set_ylim(hmin, hmax)
ax2b.set_yticks(tick_heights)
ax2b.set_yticklabels([f"{p} hPa" for p in std_levels_filtered], fontsize=9)
ax2b.set_ylabel("Pressure (hPa)", fontsize=11, color="gray")
ax2b.tick_params(axis="y", colors="gray")
 
ax2.set_xlabel("Temperature / Dew Point (°C)", fontsize=12)
ax2.set_ylabel("Geopotential Height (km)", fontsize=12)
ax2.set_ylim(hmin, hmax)
ax2.set_title(f"Vertical Profile — {STATION}\n{DATE_STR}", fontsize=13, fontweight="bold")
ax2.legend(loc="upper right", framealpha=0.9)
 
fig2.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "profile_T_Td_vs_height.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved → {out2}")
plt.close(fig2)
 
 
# ═══════════════════════════════════════════════
# FIGURE 3 — Multi-panel: T | Td | RH vs HEIGHT
# ═══════════════════════════════════════════════
fig3 = plt.figure(figsize=(13, 9))
gs   = GridSpec(1, 3, figure=fig3, wspace=0.35)
 
h_km = df["hght"] / 1000
 
# Panel 1: Temperature
ax_t = fig3.add_subplot(gs[0])
ax_t.plot(df["temp"], h_km, color=COLOR_TEMP)
ax_t.axvline(0, color="gray", lw=1.0, ls=":", alpha=0.7)
ax_t.set_xlabel("Temperature (°C)", fontsize=11)
ax_t.set_ylabel("Geopotential Height (km)", fontsize=11)
ax_t.set_ylim(hmin, hmax)
ax_t.set_title("Temperature", fontweight="bold")
 
# Panel 2: Dew Point
ax_d = fig3.add_subplot(gs[1])
ax_d.plot(df["dwpt"], h_km, color=COLOR_DWPT, ls="--")
ax_d.axvline(0, color="gray", lw=1.0, ls=":", alpha=0.7)
ax_d.set_xlabel("Dew Point (°C)", fontsize=11)
ax_d.set_ylim(hmin, hmax)
ax_d.set_yticklabels([])
ax_d.set_title("Dew Point", fontweight="bold")
 
# Panel 3: Relative Humidity
ax_r = fig3.add_subplot(gs[2])
if "relh" in df.columns and df["relh"].notna().sum() > 10:
    ax_r.plot(df["relh"], h_km, color=COLOR_RH)
    ax_r.set_xlim(0, 105)
    ax_r.axvline(100, color="gray", lw=0.8, ls=":", alpha=0.6)
ax_r.set_xlabel("Relative Humidity (%)", fontsize=11)
ax_r.set_ylim(hmin, hmax)
ax_r.set_yticklabels([])
ax_r.set_title("Relative Humidity", fontweight="bold")
 
# Shared pressure ticks on right side
ax_r2 = ax_r.twinx()
ax_r2.set_ylim(hmin, hmax)
ax_r2.set_yticks(tick_heights)
ax_r2.set_yticklabels([f"{p} hPa" for p in std_levels_filtered], fontsize=8)
ax_r2.set_ylabel("Pressure (hPa)", fontsize=10, color="gray")
ax_r2.tick_params(axis="y", colors="gray")
 
fig3.suptitle(f"Atmospheric Sounding — {STATION} | {DATE_STR}",
              fontsize=14, fontweight="bold", y=1.01)
 
fig3.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "profile_multipanel.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved → {out3}")
plt.close(fig3)
 
print("\nAll figures saved successfully.")
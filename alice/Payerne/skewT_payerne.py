import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.optimize import brentq


# ─────────────────────────────────────────────────────────────
# 0.  PATHS
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "uwyo_06610_2026-02-05_12Z.csv")
OUT_PATH   = os.path.join(SCRIPT_DIR, "skewT_payerne.png")

# ─────────────────────────────────────────────────────────────
# 1.  LOAD & FILTER DATA
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()
df = df.rename(columns={
    "pressure_hPa"             : "pres",
    "geopotential height_m"    : "hght",
    "temperature_C"            : "temp",
    "dew point temperature_C"  : "dwpt",
    "relative humidity_%"      : "relh",
    "wind direction_degree"    : "wdir",
    "wind speed_m/s"           : "wspd",
})
for col in ["pres", "hght", "temp", "dwpt", "relh", "wdir", "wspd"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
 
df = (df.dropna(subset=["pres", "hght", "temp", "dwpt"])
        .sort_values("pres", ascending=False)   # surface first
        .reset_index(drop=True))
 
# ── keep only surface → 800 hPa ──────────────────────────────
df = df[df["pres"] >= 800].reset_index(drop=True)
print(f"Levels loaded: {len(df)}  |  "
      f"{df['pres'].max():.0f} → {df['pres'].min():.0f} hPa")
 
# ─────────────────────────────────────────────────────────────
# 2.  SKEW-T GEOMETRY
# ─────────────────────────────────────────────────────────────
SKEW  = 45        # skew angle (degrees)
P_BOT = 940.0     # bottom pressure (hPa)
P_TOP = 798.0     # top    pressure (hPa)
T_MIN = -25.0     # temperature range at P_BOT
T_MAX =  15.0
 
def sx(T, P):
    """Temperature → skewed x-coordinate."""
    return T - np.tan(np.radians(SKEW)) * np.log(P / P_BOT)
 
def sy(P):
    """Pressure → y-coordinate (log scale, positive upward)."""
    return -np.log(P)
 
P_fine = np.linspace(P_BOT, P_TOP, 500)
 
x_left  = sx(T_MIN, P_BOT)
x_right = sx(T_MAX, P_BOT)
 
# ─────────────────────────────────────────────────────────────
# 3.  THERMODYNAMIC HELPERS
# ─────────────────────────────────────────────────────────────
def sat_mixing_ratio(T_C, P):
    """Saturation mixing ratio [g/kg]."""
    e_s = 6.112 * np.exp(17.67 * T_C / (T_C + 243.5))
    return 622.0 * e_s / (P - e_s)
 
def moist_adiabat(T0_C, P_arr):
    """Follow a moist adiabat upward from T0_C at P_arr[0]."""
    T   = T0_C + 273.15
    out = []
    for i, P in enumerate(P_arr):
        out.append(T - 273.15)
        if i < len(P_arr) - 1:
            dP   = P_arr[i + 1] - P
            Lv   = 2.5e6;  eps = 0.622
            rs   = sat_mixing_ratio(T - 273.15, P) / 1000
            gd   = 9.8e-3
            num  = gd * (1 + Lv * rs / (287 * T))
            den  = 1 + Lv**2 * rs * eps / (1004 * 287 * T**2)
            dTdP = (num / den) / (P * 100 / 287 / T)
            T   += dTdP * dP
    return np.array(out)
 
# ─────────────────────────────────────────────────────────────
# 4.  FIGURE
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 11))
ax.set_xlim(x_left - 0.5, x_right + 5.5)
ax.set_ylim(sy(P_BOT) - 0.005, sy(P_TOP) + 0.015)
ax.set_facecolor("#f8f8f8")
ax.set_xticks([]);  ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
 
# ── dry adiabats (θ = const) ──────────────────────────────────
for th in np.arange(265, 320, 5):
    T_dry = th * (P_fine / 1000) ** 0.286 - 273.15
    ax.plot([sx(t, p) for t, p in zip(T_dry, P_fine)],
            [sy(p)    for p in P_fine],
            color="#c8a060", lw=0.65, alpha=0.55, zorder=1)
 
# ── moist adiabats ────────────────────────────────────────────
for T0 in np.arange(-5, 20, 3):
    Tm = moist_adiabat(T0, P_fine)
    ax.plot([sx(t, p) for t, p in zip(Tm, P_fine)],
            [sy(p)    for p in P_fine],
            color="#60a8c8", lw=0.65, alpha=0.55, zorder=1)
 
# ── saturation mixing ratio lines ────────────────────────────
for w in [1, 2, 4, 7, 10, 16]:
    T_w = []
    for p in P_fine:
        try:
            T_w.append(brentq(lambda T: sat_mixing_ratio(T, p) - w, -40, 50))
        except Exception:
            T_w.append(np.nan)
    T_w  = np.array(T_w)
    mask = ~np.isnan(T_w)
    if mask.sum() > 5:
        ax.plot(np.array([sx(t, p) for t, p in zip(T_w, P_fine)])[mask],
                np.array([sy(p) for p in P_fine])[mask],
                color="#4aaa4a", lw=0.6, ls="--", alpha=0.5, zorder=1)
        idx = mask.nonzero()[0][-1]
        ax.text(sx(T_w[idx], P_fine[idx]), sy(P_fine[idx]) + 0.001,
                f"{w}", color="#2a7a2a", fontsize=7.5,
                ha="center", va="bottom", zorder=3)
 
# ── isotherms ─────────────────────────────────────────────────
for T_iso in np.arange(-40, 41, 5):
    col = "#aaaaaa"
    lw  = 0.55
    ax.plot([sx(T_iso, p) for p in P_fine],
            [sy(p)        for p in P_fine],
            color=col, lw=lw, alpha=0.7, zorder=1)
    xt = sx(T_iso, P_BOT)
    if x_left - 0.3 <= xt <= x_right + 0.3:
        ax.text(xt, sy(P_BOT) - 0.003, f"{T_iso}°",
                fontsize=7.5, color=col, ha="center", va="top", zorder=3)
 
# ── isobar lines + pressure labels ───────────────────────────
for P_iso in [950, 925, 900, 875, 850, 825, 800]:
    if P_TOP <= P_iso <= P_BOT:
        ax.axhline(sy(P_iso), color="#888888", lw=0.6, alpha=0.45, zorder=1)
        ax.text(x_left - 0.4, sy(P_iso), f"{P_iso}",
                fontsize=9, ha="right", va="center",
                color="#444444", fontweight="bold")
 
# ── Temperature & Dew Point ───────────────────────────────────
xs_T  = [sx(t, p) for t, p in zip(df["temp"], df["pres"])]
xs_Td = [sx(t, p) for t, p in zip(df["dwpt"], df["pres"])]
ys    = [sy(p)    for p in df["pres"]]
 
ax.plot(xs_T,  ys, color="#cc2222", lw=2.5, label="Temperature (°C)", zorder=5)
ax.plot(xs_Td, ys, color="#1155cc", lw=2.5, label="Dew Point (°C)",
        linestyle="--", zorder=5)
 
# ── Wind barbs ────────────────────────────────────────────────
x_barb = x_right + 3.2
step   = max(1, len(df) // 18)
for _, row in df.iloc[::step].iterrows():
    if pd.notna(row["wdir"]) and pd.notna(row["wspd"]) and row["wspd"] > 0.1:
        u = -row["wspd"] * np.sin(np.radians(row["wdir"]))
        v = -row["wspd"] * np.cos(np.radians(row["wdir"]))
        ax.barbs(x_barb, sy(row["pres"]), u, v,
                 length=7, linewidth=1.0,
                 barbcolor="#222222", flagcolor="#222222", zorder=6)
 
ax.text(x_barb, sy(P_TOP) + 0.008, "Wind\n(m/s)",
        fontsize=8, ha="center", va="bottom", color="#333333")
 
# ── Legend ────────────────────────────────────────────────────
handles = [
    plt.Line2D([0],[0], color="#cc2222", lw=2.2,             label="Temperature (°C)"),
    plt.Line2D([0],[0], color="#1155cc", lw=2.2, ls="--",    label="Dew Point (°C)"),
    plt.Line2D([0],[0], color="#c8a060", lw=1.2, alpha=0.8,  label="Dry adiabats"),
    plt.Line2D([0],[0], color="#60a8c8", lw=1.2, alpha=0.8,  label="Moist adiabats"),
    plt.Line2D([0],[0], color="#4aaa4a", lw=1.0, ls="--",    label="Sat. mixing ratio (g/kg)"),
    plt.Line2D([0],[0], color="#aaaaaa", lw=0.9,             label="Isotherms"),
]
ax.legend(handles=handles, loc="upper left", fontsize=9,
          framealpha=0.93, edgecolor="#cccccc")
 
# ── Pressure axis label ───────────────────────────────────────
ax.text(x_left - 1.8, (sy(P_BOT) + sy(P_TOP)) / 2,
        "Pressure (hPa)", fontsize=10, rotation=90,
        ha="center", va="center", color="#444444")
 
# ── Title ─────────────────────────────────────────────────────
ax.set_title(
    "Skew-T / log-P  —  Payerne (06610)\n"
    "2026-02-05  12 UTC  |  surface → 800 hPa",
    fontsize=13, fontweight="bold", pad=14
)
 
# ─────────────────────────────────────────────────────────────
# 5.  SAVE
# ─────────────────────────────────────────────────────────────
plt.tight_layout(pad=1.5)
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT_PATH}")
 
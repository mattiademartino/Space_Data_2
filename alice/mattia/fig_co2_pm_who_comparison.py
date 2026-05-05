"""
Single figure for VAMOS CO2 and ScamSat particulate matter profiles.
 
Outputs:
  alice/mattia/fig_co2_pm_who_comparison.png
"""
 
from __future__ import annotations
 
import os
import sys
from pathlib import Path
 
 
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(os.getcwd()) / "alice" / "mattia"
ROOT_DIR = SCRIPT_DIR.parents[1]
 
# Matplotlib tries to write its cache under ~/.config on this machine.
MPL_CACHE_DIR = Path("/tmp") / "space_data_2_matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
 
sys.path.insert(0, str(ROOT_DIR))
sys.dont_write_bytecode = True
os.chdir(ROOT_DIR)
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
from utils import (
    WHO_PM10_24H,
    WHO_PM25_24H,
    detect_vamos_drop,
    load_scamsat_bundle,
    load_vamos_science,
)
 
 
# Update this once the real background CO2 reference is available.
CO2_BACKGROUND_PPM = 420.0
 
# ScamSat descent interval used in the existing analysis notebooks/scripts.
SCAMSAT_DROP_START_S = 828.0
SCAMSAT_DROP_END_S = 996.0
 
# NABEL ground reference – Zürich-Kaserne, 14:30 local time.
NABEL_PM25 = 17.29   # µg m⁻³
NABEL_PM10 = 26.07   # µg m⁻³
 
OUTPUT_PATH = SCRIPT_DIR / "fig_co2_pm_who_comparison.png"
 
 
def binned_median_profile(
    data: pd.DataFrame,
    value_col: str,
    altitude_col: str,
    *,
    n_bins: int = 36,
) -> pd.DataFrame:
    """Return a smooth median vertical profile without hiding raw samples."""
    profile = data[[value_col, altitude_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if profile.empty:
        return profile
 
    z_min = float(profile[altitude_col].min())
    z_max = float(profile[altitude_col].max())
    if np.isclose(z_min, z_max):
        return profile.sort_values(altitude_col)
 
    bins = np.linspace(z_min, z_max, n_bins + 1)
    profile = profile.assign(_bin=pd.cut(profile[altitude_col], bins, include_lowest=True))
    med = (
        profile.groupby("_bin", observed=True)
        .agg({value_col: "median", altitude_col: "median"})
        .dropna()
        .reset_index(drop=True)
        .sort_values(altitude_col)
    )
    return med
 
 
def load_vamos_co2_profile() -> pd.DataFrame:
    """Load VAMOS CO2 during descent and attach altitude AGL."""
    vamos = load_vamos_science()
    drop = detect_vamos_drop(vamos)
 
    co2 = vamos.loc[drop["drop_mask"], ["co2_ppm"]].copy()
    co2["altitude_agl_m"] = np.asarray(drop["h_agl"])[drop["drop_mask"]]
    co2 = co2.replace([np.inf, -np.inf], np.nan).dropna()
    co2 = co2[co2["co2_ppm"] > 0].reset_index(drop=True)
    return co2
 
 
def load_scamsat_pm_profile(key: str, value_col: str) -> pd.DataFrame:
    """Load one ScamSat PM channel during the descent interval."""
    bundle = load_scamsat_bundle()
    frame = bundle[key].copy()
    mask = frame["t_s"].between(SCAMSAT_DROP_START_S, SCAMSAT_DROP_END_S)
    pm = frame.loc[mask, [value_col, "altitude_agl_m"]].copy()
    pm = pm.replace([np.inf, -np.inf], np.nan).dropna()
    pm = pm[pm[value_col] >= 0].reset_index(drop=True)
    return pm
 
 
def add_vertical_reference(
    ax: plt.Axes,
    x_value: float,
    label: str,
    *,
    color: str,
    linestyle: str = "--",
) -> None:
    """Draw a labelled vertical reference line."""
    ax.axvline(x_value, color=color, linestyle=linestyle, linewidth=1.9, alpha=0.95)
    y_top = ax.get_ylim()[1]
    ax.text(
        x_value,
        y_top,
        f" {label}",
        rotation=90,
        va="top",
        ha="left",
        color=color,
        fontsize=8.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )
 
 
def make_plot() -> Path:
    """Create and save the combined CO2/PM figure."""
    co2 = load_vamos_co2_profile()
    pm25 = load_scamsat_pm_profile("pm25", "pm25")
    pm10 = load_scamsat_pm_profile("pm10", "pm10")
 
    co2_med = binned_median_profile(co2, "co2_ppm", "altitude_agl_m")
    pm25_med = binned_median_profile(pm25, "pm25", "altitude_agl_m")
    pm10_med = binned_median_profile(pm10, "pm10", "altitude_agl_m")
 
    all_alt = pd.concat(
        [co2["altitude_agl_m"], pm25["altitude_agl_m"], pm10["altitude_agl_m"]],
        ignore_index=True,
    ).dropna()
    z_min = max(0.0, float(all_alt.min()) - 20.0)
    z_max = float(all_alt.max()) + 30.0
    z_range = z_max - z_min
 
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
 
    fig, (ax_co2, ax_pm) = plt.subplots(
        1,
        2,
        figsize=(10.0, 8.0),
        sharey=True,
        constrained_layout=True,
    )
 
    ax_co2.scatter(
        co2["co2_ppm"],
        co2["altitude_agl_m"],
        s=15,
        color="#2563eb",
        alpha=0.28,
        edgecolors="none",
        label="VAMOS samples",
    )
    ax_co2.plot(
        co2_med["co2_ppm"],
        co2_med["altitude_agl_m"],
        color="#1d4ed8",
        linewidth=2.4,
        label="VAMOS binned median",
    )
    ax_co2.set_ylim(z_min, z_max)
    add_vertical_reference(
        ax_co2,
        CO2_BACKGROUND_PPM,
        f"background CO2 = {CO2_BACKGROUND_PPM:.0f} ppm",
        color="#4b5563",
    )
    ax_co2.set_xlabel("CO2 [ppm]")
    ax_co2.set_ylabel("Altitude AGL [m]")
    ax_co2.set_title("VAMOS CO2 profile")
    ax_co2.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=8)
 
    ax_pm.scatter(
        pm25["pm25"],
        pm25["altitude_agl_m"],
        s=15,
        color="#f97316",
        alpha=0.26,
        edgecolors="none",
        label="ScamSat PM2.5 samples",
    )
    ax_pm.scatter(
        pm10["pm10"],
        pm10["altitude_agl_m"],
        s=15,
        color="#16a34a",
        alpha=0.22,
        edgecolors="none",
        label="ScamSat PM10 samples",
    )
    ax_pm.plot(
        pm25_med["pm25"],
        pm25_med["altitude_agl_m"],
        color="#ea580c",
        linewidth=2.4,
        label="PM2.5 binned median",
    )
    ax_pm.plot(
        pm10_med["pm10"],
        pm10_med["altitude_agl_m"],
        color="#15803d",
        linewidth=2.4,
        linestyle="--",
        label="PM10 binned median",
    )
 
    pm_max = float(pd.concat([pm25["pm25"], pm10["pm10"]]).max())
    ax_pm.set_xlim(left=0, right=max(pm_max * 1.12, WHO_PM10_24H * 1.25))
    ax_pm.axvspan(WHO_PM25_24H, ax_pm.get_xlim()[1], color="#f97316", alpha=0.045)
    ax_pm.axvspan(WHO_PM10_24H, ax_pm.get_xlim()[1], color="#15803d", alpha=0.055)
    
    add_vertical_reference(
        ax_pm,
        WHO_PM25_24H,
        f"WHO PM2.5 24 h = {WHO_PM25_24H:g}",
        color="#c2410c",
        linestyle=":",
    )
    add_vertical_reference(
        ax_pm,
        WHO_PM10_24H,
        f"WHO PM10 24 h = {WHO_PM10_24H:g}",
        color="#15803d",
        linestyle=":",
    )
 
    # ── NABEL ground reference – Zürich-Kaserne, 14:30 ───────────────────
    # Stagger the two markers vertically so they don't overlap.
    nabel_alt_pm25 = z_min                      # PM2.5 sits at ground
    nabel_alt_pm10 = z_min + z_range * 0.08     # PM10 sits ~8 % higher
 
    ax_pm.scatter(
        [NABEL_PM25], [nabel_alt_pm25],
        s=130, color="#ea580c", marker="D", zorder=6,
        edgecolors="white", linewidths=0.8,
        label=f"NABEL PM2.5 ground = {NABEL_PM25} µg m⁻³",
    )
    ax_pm.scatter(
        [NABEL_PM10], [nabel_alt_pm10],
        s=130, color="#15803d", marker="D", zorder=6,
        edgecolors="white", linewidths=0.8,
        label=f"NABEL PM10 ground = {NABEL_PM10} µg m⁻³",
    )
 
    ax_pm.set_xlabel("Particulate matter [microg m$^{-3}$]")
    ax_pm.set_title("ScamSat particulate matter profile")
    ax_pm.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=8)
 
    fig.suptitle(
        "CO2 and particulate matter during CanSat descent",
        fontsize=13,
        fontweight="bold",
    )
 
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
 
    co2_delta = co2["co2_ppm"].median() - CO2_BACKGROUND_PPM
    print(f"Saved: {OUTPUT_PATH}")
    print(f"VAMOS CO2 median during descent: {co2['co2_ppm'].median():.1f} ppm")
    print(f"CO2 median minus background: {co2_delta:+.1f} ppm")
    print(f"ScamSat PM2.5 median during descent: {pm25['pm25'].median():.1f} microg/m3")
    print(f"ScamSat PM10 median during descent: {pm10['pm10'].median():.1f} microg/m3")
    print(f"NABEL PM2.5 ground reference (Zürich-Kaserne 14:30): {NABEL_PM25} µg/m3")
    print(f"NABEL PM10 ground reference (Zürich-Kaserne 14:30): {NABEL_PM10} µg/m3")
    return OUTPUT_PATH
 
 
if __name__ == "__main__":
    make_plot()
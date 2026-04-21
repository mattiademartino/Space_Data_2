"""
CanSat Spectral Analysis - Complete Reproduction Script
========================================================

This script reproduces all 6 figures from the time-frequency analysis report.

Data requirements:
  - science_VAMOS.csv
  - wind_VAMOS.csv
  - science_GRASP.csv
  - OBAMA_data_decoded.xlsx

Output:
  - 6 PNG figures saved to ./figures/
  - summary.json with key parameters

Author: [Your Name]
Date: April 2026
Course: Space System Engineering, ETH Zürich
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram, get_window
import json
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths (adjust if needed)
DATA_DIR = "."  # directory containing CSV/XLSX files
OUTPUT_DIR = "./figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ISA (International Standard Atmosphere) constants
T0_ISA = 288.15   # K at sea level
P0_ISA = 1013.25  # hPa
L_ISA = 0.0065    # K/m lapse rate
g = 9.80665       # m/s²
R = 287.05        # J/(kg·K)

# Plotting style
plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def isa_temperature(h):
    """ISA temperature [K] at altitude h [m]."""
    return T0_ISA - L_ISA * h

def isa_pressure(h):
    """ISA pressure [hPa] at altitude h [m]."""
    T_h = isa_temperature(h)
    return P0_ISA * (T_h / T0_ISA) ** (g / (R * L_ISA))

def barometric_altitude(p, p0):
    """
    Compute barometric altitude [m] from pressure using standard formula.
    
    Parameters:
        p: measured pressure [hPa]
        p0: reference ground pressure [hPa]
    """
    return 44330 * (1 - (p / p0) ** (1/5.255))

def resample_uniform(t, y, fs_new):
    """Resample signal y(t) onto uniform grid at fs_new Hz using linear interpolation."""
    t0, t1 = t[0], t[-1]
    tu = np.arange(t0, t1, 1.0/fs_new)
    yu = np.interp(tu, t, y)
    return tu, yu

def clean_numeric(df, cols):
    """Coerce columns to numeric, drop NaN rows."""
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=cols).reset_index(drop=True)

# ============================================================================
# DATA LOADING
# ============================================================================

print("Loading data...")

# --- VAMOS science ---
sci_v = pd.read_csv(f"{DATA_DIR}/science_VAMOS.csv")
sci_v = clean_numeric(sci_v, sci_v.columns.tolist())
t_v = sci_v["timestamp_ms"].values / 1000.0
p_v = sci_v["pressure_hPa"].values
T_v = sci_v["temperature_C"].values
co2_v = sci_v["co2_ppm"].values

# --- VAMOS wind ---
wind_v = pd.read_csv(f"{DATA_DIR}/wind_VAMOS.csv")
wind_v = clean_numeric(wind_v, wind_v.columns.tolist())
tw_v = wind_v["timestamp_ms"].values / 1000.0
ax_v = wind_v["x_wind_acc"].values
ay_v = wind_v["y_wind_acc"].values
ws_v = wind_v["wind_speed"].values

# Wind data has timestamp resets during ground tests → keep only monotonic tail
dt_w = np.diff(tw_v)
reset_idx = np.where(dt_w < 0)[0]
if len(reset_idx):
    last_reset = reset_idx[-1] + 1
    tw_v = tw_v[last_reset:]
    ax_v = ax_v[last_reset:]
    ay_v = ay_v[last_reset:]
    ws_v = ws_v[last_reset:]
    print(f"  VAMOS wind: removed {last_reset} pre-reset samples, {len(tw_v)} remain")

# --- GRASP ---
raw_g = open(f"{DATA_DIR}/science_GRASP.csv").read().splitlines()
header_g = [c.strip() for c in raw_g[0].split(",")]
data_rows_g = []
for line in raw_g[1:]:
    parts = line.split(",")
    try:
        float(parts[0])
        data_rows_g.append(parts)
    except ValueError:
        pass  # skip event rows
        
sci_g = pd.DataFrame(data_rows_g, columns=header_g)
sci_g = clean_numeric(sci_g, sci_g.columns.tolist())

# Drop first sample if it's a power-on outlier
if sci_g["pressure [Pa]"].iloc[0] < 850 * 100:
    sci_g = sci_g.iloc[1:].reset_index(drop=True)
    
t_g = sci_g["timestamp [ms]"].values / 1000.0
p_g = sci_g["pressure [Pa]"].values / 100.0  # to hPa
T_g = sci_g["temperature [C]"].values
alt_g = sci_g["altitude [m]"].values
pm25_g = sci_g["pm2_5 [microg/m3]"].values
pm10_g = sci_g["pm10_0 [microg/m3]"].values

# --- OBAMA ---
obm = pd.read_excel(f"{DATA_DIR}/OBAMA_data_decoded.xlsx", sheet_name="decoded")
for col in ["Time_s", "first_pstat_avg_hPa", "first_temp_avg_C", "first_hum_avg_pct"]:
    obm[col] = pd.to_numeric(obm[col], errors="coerce")
obm = obm[(obm["Time_s"] > 0) & (obm["Time_s"] < 500) 
          & (obm["first_temp_avg_C"].between(-50, 60))].reset_index(drop=True)
obm = obm.sort_values("Time_s").reset_index(drop=True)
t_o = obm["Time_s"].values
p_o = obm["first_pstat_avg_hPa"].values
T_o = obm["first_temp_avg_C"].values
h_o = obm["first_hum_avg_pct"].values

print(f"  VAMOS: {len(sci_v)} science samples, {len(tw_v)} wind samples")
print(f"  GRASP: {len(sci_g)} samples")
print(f"  OBAMA: {len(obm)} samples")

# ============================================================================
# DROP IDENTIFICATION (DATA-DRIVEN)
# ============================================================================

print("\nIdentifying VAMOS drop phase...")

# Ground pressure from pre/post flight
p0_vamos = (np.median(p_v[:100]) + np.median(p_v[-500:])) / 2
print(f"  Ground pressure p0 = {p0_vamos:.2f} hPa")

# Compute barometric altitude
h_vamos = barometric_altitude(p_v, p0_vamos)

# Identify drop via largest dp/dt block
p_smooth = pd.Series(p_v).rolling(5, center=True, min_periods=1).median().values
dpdt = np.gradient(p_smooth) / np.gradient(t_v)

# Find contiguous blocks where dp/dt > 0.1 hPa/s
above = dpdt > 0.1
idx = np.where(above)[0]
if len(idx) == 0:
    raise ValueError("No drop phase detected (no dp/dt > 0.1 hPa/s)")
    
gaps = np.where(np.diff(idx) > 15)[0]
if len(gaps):
    starts = np.concatenate([[idx[0]], idx[gaps+1]])
    ends = np.concatenate([idx[gaps], [idx[-1]]])
else:
    starts = np.array([idx[0]])
    ends = np.array([idx[-1]])
    
# Select block with largest Δp
dps = np.array([p_v[e] - p_v[s] for s, e in zip(starts, ends)])
i_best = np.argmax(dps)
ds, de = starts[i_best], ends[i_best]

# Refine: apogee = pressure minimum before this block
look_back = max(0, ds - 60)
apogee_idx = look_back + int(np.argmin(p_v[look_back:ds+1]))

# Landing = when pressure returns to p0
look_fwd = min(len(p_v), de + 60)
post_ground = np.where(p_v[de:look_fwd] > p0_vamos - 0.5)[0]
landing_idx = de + (post_ground[0] if len(post_ground) else 0)

t_apogee = t_v[apogee_idx]
t_landing = t_v[landing_idx]
h_peak = h_vamos[apogee_idx]
drop_duration = t_landing - t_apogee
descent_rate = (h_vamos[apogee_idx] - h_vamos[landing_idx]) / drop_duration

print(f"  Apogee: t={t_apogee:.0f} s, h={h_peak:.0f} m AGL, p={p_v[apogee_idx]:.1f} hPa")
print(f"  Landing: t={t_landing:.0f} s")
print(f"  Drop duration: {drop_duration:.0f} s")
print(f"  Mean descent rate: {descent_rate:.2f} m/s")

# Save summary
summary = {
    "p0_vamos": float(p0_vamos),
    "t_apogee": float(t_apogee),
    "t_landing": float(t_landing),
    "h_peak_m": float(h_peak),
    "drop_duration_s": float(drop_duration),
    "descent_rate_m_per_s": float(descent_rate),
}
with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ============================================================================
# FIGURE 1: Three Groups Overview
# ============================================================================

print("\nGenerating Figure 1: Three groups overview...")

fig, axes = plt.subplots(3, 1, figsize=(10, 7))

# VAMOS
axes[0].plot(t_v, p_v, color="C0", lw=1)
axes[0].set_ylabel("p [hPa]")
axes[0].set_title("VAMOS  (fs ≈ 1 Hz,  N=5945)")
axes[0].set_xlabel("t [s, onboard clock]")

# GRASP
axes[1].plot(t_g - 2521.545, p_g, color="C2", lw=1)
axes[1].set_ylabel("p [hPa]")
axes[1].set_title("GRASP  (fs ≈ 38 Hz,  N=1715) — aligned to 'freefall' trigger")
axes[1].set_xlabel("t from trigger [s]")

# OBAMA
axes[2].plot(t_o, p_o, "o-", color="C4", lw=1, markersize=4)
axes[2].set_ylabel("p [hPa]")
axes[2].set_title("OBAMA  (dt ≈ 25 s,  N=21) — coarse quantization")
axes[2].set_xlabel("t [s, onboard clock]")

plt.suptitle("Three groups, three sensors, three sampling regimes", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_three_groups.png", dpi=130, bbox_inches="tight")
plt.close()

# ============================================================================
# FIGURE 2: VAMOS Drop Identification
# ============================================================================

print("Generating Figure 2: VAMOS drop identification...")

fig, (ax_p, ax_h) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

ax_p.plot(t_v, p_v, color="C0", lw=1)
ax_p.axhline(p0_vamos, color="gray", ls=":", lw=1, 
             label=f"$p_0$={p0_vamos:.1f} hPa (pre/post-flight median)")
ax_p.axvspan(t_apogee, t_landing, color="orange", alpha=0.25, label=f"drop (data-driven)")
ax_p.legend(loc="lower right", fontsize=9)
ax_p.set_ylabel("pressure [hPa]")
ax_p.set_title("VAMOS pressure → drop phase identified from data, not from rulebook")

ax_h.plot(t_v, h_vamos, color="C2", lw=1)
ax_h.axhline(h_peak, color="C2", ls=":", lw=1, label=f"peak = {h_peak:.0f} m AGL")
ax_h.axvspan(t_apogee, t_landing, color="orange", alpha=0.25)
ax_h.legend(loc="upper right", fontsize=9)
ax_h.set_ylabel("barometric altitude [m AGL]")
ax_h.set_xlabel("t [s]")
ax_h.text(0.02, 0.95,
          f"Drop duration: {drop_duration:.0f} s\n"
          f"Mean descent rate: {descent_rate:.2f} m/s\n"
          f"Peak altitude: {h_peak:.0f} m AGL",
          transform=ax_h.transAxes, va="top",
          bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_vamos_drop.png", dpi=130, bbox_inches="tight")
plt.close()

# ============================================================================
# FIGURE 3: Welch PSD for 3 Phases
# ============================================================================

print("Generating Figure 3: Welch PSD across phases...")

# Define phase masks
in_plane_mask = (p_v < p0_vamos - 30)
in_plane_idx = np.where(in_plane_mask)[0]
if len(in_plane_idx):
    t_plane_start = t_v[in_plane_idx[0]] + 60
    t_plane_end = t_apogee - 30
else:
    t_plane_start = t_v[0]
    t_plane_end = t_apogee

mask_aircraft = (tw_v > t_plane_start) & (tw_v < t_plane_end)
mask_drop = (tw_v >= t_apogee) & (tw_v <= t_landing)
mask_ground = (tw_v > t_landing + 30)

phases = {
    "aircraft (in plane, cruise)": (mask_aircraft, "C0"),
    "drop (under parachute)": (mask_drop, "C3"),
    "ground (after landing)": (mask_ground, "gray"),
}

# Welch parameters
fs = 1.0
nperseg = 64

def compute_welch_psd(t, y, fs, nperseg, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2
    tu, yu = resample_uniform(t, y, fs)
    yu = yu - np.mean(yu)
    nperseg_actual = min(nperseg, len(yu))
    noverlap_actual = min(noverlap, nperseg_actual - 1)
    f, P = welch(yu, fs=fs, window="hann", nperseg=nperseg_actual,
                 noverlap=noverlap_actual, detrend="linear", scaling="density")
    dof = 2 * (len(yu) / nperseg_actual) * 2.2  # Hann effective DoF
    return f, P, len(yu), dof

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for sig_name, sig, ax in [("|wind accel| = sqrt(ax²+ay²)", np.sqrt(ax_v**2 + ay_v**2), axes[0]),
                          ("wind speed", ws_v, axes[1])]:
    for name, (m, c) in phases.items():
        if m.sum() < nperseg:
            continue
        f, P, Nu, dof = compute_welch_psd(tw_v[m], sig[m], fs, nperseg)
        ax.loglog(f[1:], P[1:], color=c, label=f"{name}  (N={m.sum()}, DoF≈{dof:.0f})", lw=1.4)
    
    ax.set_xlabel("frequency [Hz]")
    ylabel = "PSD [(m/s²)²/Hz]" if "accel" in sig_name else "PSD [(m/s)²/Hz]"
    ax.set_ylabel(ylabel)
    ax.set_title(f"Welch PSD — {sig_name}")
    ax.legend(fontsize=8, loc="lower left")
    ax.axvline(fs/2, color="red", ls="--", lw=0.8, alpha=0.6)
    ax.text(fs/2, ax.get_ylim()[1], "Nyquist", color="red", ha="right", va="top", fontsize=8)

plt.suptitle(f"VAMOS Welch PSD across flight phases  "
             f"(fs={fs} Hz, Hann, nperseg={nperseg}, 50% overlap, linear detrend)",
             fontsize=10, y=1.03)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_welch_psd.png", dpi=130, bbox_inches="tight")
plt.close()

# ============================================================================
# FIGURE 4: VAMOS Spectrogram
# ============================================================================

print("Generating Figure 4: VAMOS spectrogram...")

# Resample wind acceleration magnitude to uniform 1 Hz
acc_mag = np.sqrt(ax_v**2 + ay_v**2)
tu, accu = resample_uniform(tw_v, acc_mag, fs)
accu = accu - np.mean(accu)

# Compute spectrogram
NPS = 64
NOV = 56
f_sp, t_sp, S = spectrogram(accu, fs=fs, window="hann", nperseg=NPS, 
                            noverlap=NOV, scaling="density")
t_sp_abs = t_sp + tu[0]
S_dB = 10 * np.log10(S + 1e-12)

# Plot
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure(figsize=(11, 5.8))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.05)
ax_p = fig.add_subplot(gs[0])
ax_s = fig.add_subplot(gs[1], sharex=ax_p)

ax_p.plot(t_v, p_v, color="C0", lw=1)
ax_p.axvline(t_apogee, color="r", ls="--", lw=1, label=f"apogee (t={t_apogee:.0f}s)")
ax_p.axvline(t_landing, color="g", ls="--", lw=1, label=f"landing (t={t_landing:.0f}s)")
ax_p.legend(loc="lower right", fontsize=8)
ax_p.set_ylabel("pressure [hPa]")
ax_p.set_title(f"VAMOS spectrogram of |wind accel|  "
               f"(fs={fs} Hz, Hann, nperseg={NPS} → Δt={NPS/fs:.0f}s, "
               f"Δf={fs/NPS:.3f} Hz, overlap={NOV/NPS*100:.0f}%)")
ax_p.tick_params(labelbottom=False)

pcm = ax_s.pcolormesh(t_sp_abs, f_sp, S_dB, shading="auto", cmap="viridis",
                      vmin=np.nanpercentile(S_dB, 5), vmax=np.nanpercentile(S_dB, 99))
ax_s.axvline(t_apogee, color="r", ls="--", lw=1)
ax_s.axvline(t_landing, color="g", ls="--", lw=1)
ax_s.set_ylabel("f [Hz]")
ax_s.set_xlabel("t [s, onboard clock]")

# Add colorbar with matching width
divider_s = make_axes_locatable(ax_s)
cax_s = divider_s.append_axes("right", size="2%", pad=0.1)
plt.colorbar(pcm, cax=cax_s, label="PSD [dB re (m/s²)²/Hz]")
divider_p = make_axes_locatable(ax_p)
cax_p = divider_p.append_axes("right", size="2%", pad=0.1)
cax_p.set_visible(False)

# Phase labels
for t0, t1, label in [(tu[0], t_apogee-30, "pre-apogee"),
                      (t_apogee, t_landing, "DROP"),
                      (t_landing+30, tu[-1], "post-landing")]:
    if t1 > t0:
        ax_s.annotate(label, xy=((t0+t1)/2, 0.47), ha="center", va="top",
                     fontsize=9, color="white",
                     bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))

plt.savefig(f"{OUTPUT_DIR}/fig4_spectrogram_vamos.png", dpi=130, bbox_inches="tight")
plt.close()

# ============================================================================
# FIGURE 5: GRASP Spectral Analysis
# ============================================================================

print("Generating Figure 5: GRASP spectral analysis...")

# Resample GRASP to uniform 30 Hz
fs_g = 30.0
tu_g = np.arange(t_g[0], t_g[-1], 1.0/fs_g)
pu_g = np.interp(tu_g, t_g, p_g)

# Detrend with polynomial fit
coef = np.polyfit(tu_g - tu_g[0], pu_g, 3)
pu_g_trend = np.polyval(coef, tu_g - tu_g[0])
pu_g_res = pu_g - pu_g_trend

# Welch and spectrogram on residual
NPS_G = 1024
NOV_G = 768
f_wg, P_wg = welch(pu_g_res, fs=fs_g, window="hann", nperseg=NPS_G,
                   noverlap=NOV_G, detrend=False)
f_sg, t_sg, S_g = spectrogram(pu_g_res, fs=fs_g, window="hann",
                              nperseg=NPS_G, noverlap=NOV_G, scaling="density")

fig, axes = plt.subplots(2, 2, figsize=(12, 6))

# Pressure + trend
axes[0,0].plot(tu_g - 2521.545, pu_g, color="C2", label="p (raw resampled)")
axes[0,0].plot(tu_g - 2521.545, pu_g_trend, color="k", ls="--", lw=1, label="order-3 trend")
axes[0,0].set_xlabel("t from trigger [s]")
axes[0,0].set_ylabel("p [hPa]")
axes[0,0].set_title("GRASP pressure and detrended")
axes[0,0].legend(fontsize=8)

# Residual
axes[0,1].plot(tu_g - 2521.545, pu_g_res, color="C3", lw=0.8)
axes[0,1].set_xlabel("t from trigger [s]")
axes[0,1].set_ylabel("p residual [hPa]")
axes[0,1].set_title("Detrended pressure (residual)")

# Welch PSD
axes[1,0].loglog(f_wg[1:], P_wg[1:], color="C3", lw=1.4)
axes[1,0].set_xlabel("frequency [Hz]")
axes[1,0].set_ylabel("PSD [hPa²/Hz]")
axes[1,0].set_title(f"Welch PSD of pressure residual  (fs={fs_g} Hz, nperseg={NPS_G})")
axes[1,0].axvspan(0.2, 1.0, color="orange", alpha=0.15, label="pendulum band")
axes[1,0].axvspan(1.0, 3.0, color="red", alpha=0.15, label="spin band")
axes[1,0].legend(fontsize=8)

# Spectrogram
pcm = axes[1,1].pcolormesh(t_sg + tu_g[0] - 2521.545, f_sg,
                           10*np.log10(S_g + 1e-16), shading="auto", cmap="viridis")
axes[1,1].set_ylim(0, 3)
axes[1,1].set_xlabel("t from trigger [s]")
axes[1,1].set_ylabel("f [Hz]")
axes[1,1].set_title("Spectrogram of residual")
plt.colorbar(pcm, ax=axes[1,1], label="dB")

plt.suptitle("GRASP — high fs allows observing pendulum/spin band", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig5_grasp_spectral.png", dpi=130, bbox_inches="tight")
plt.close()

# ============================================================================
# FIGURE 6: ISA Comparison
# ============================================================================

print("Generating Figure 6: ISA comparison...")

# Generate ISA profile
h_isa_profile = np.linspace(0, 1200, 200)
T_isa_profile = isa_temperature(h_isa_profile) - 273.15  # to °C
p_isa_profile = isa_pressure(h_isa_profile)

# VAMOS drop phase
drop_mask = (t_v >= t_apogee) & (t_v <= t_landing)
h_drop = h_vamos[drop_mask]
p_drop = p_v[drop_mask]
T_drop = T_v[drop_mask]

# GRASP altitudes (using VAMOS p0)
h_g = barometric_altitude(p_g, p0_vamos)

# OBAMA altitudes
h_o_alt = barometric_altitude(p_o, p0_vamos)

fig, (ax_T, ax_p) = plt.subplots(1, 2, figsize=(11, 5))

# Temperature profile
ax_T.plot(T_isa_profile, h_isa_profile, 'k--', lw=2, label='ISA reference', zorder=1)
ax_T.scatter(T_drop, h_drop, s=8, alpha=0.4, c='C0', label=f'VAMOS drop (N={len(h_drop)})', zorder=3)
ax_T.scatter(T_g, h_g, s=8, alpha=0.4, c='C2', label=f'GRASP (N={len(h_g)})', zorder=2)
ax_T.scatter(T_o, h_o_alt, s=30, alpha=0.7, c='C4', marker='s', label=f'OBAMA (N={len(h_o_alt)})', zorder=4)
ax_T.set_xlabel('Temperature [°C]')
ax_T.set_ylabel('Altitude [m AGL]')
ax_T.set_title('Atmospheric temperature profile')
ax_T.legend(fontsize=9)
ax_T.set_ylim(0, 1000)

# Pressure profile
ax_p.plot(p_isa_profile, h_isa_profile, 'k--', lw=2, label='ISA reference', zorder=1)
ax_p.scatter(p_drop, h_drop, s=8, alpha=0.4, c='C0', label=f'VAMOS drop', zorder=3)
ax_p.scatter(p_g, h_g, s=8, alpha=0.4, c='C2', label=f'GRASP', zorder=2)
ax_p.scatter(p_o, h_o_alt, s=30, alpha=0.7, c='C4', marker='s', label=f'OBAMA', zorder=4)
ax_p.axhline(448, color='gray', ls=':', lw=1, label='Dübendorf ground (448 m ASL)')
ax_p.set_xlabel('Pressure [hPa]')
ax_p.set_ylabel('Altitude [m AGL]')
ax_p.set_title('Atmospheric pressure profile')
ax_p.legend(fontsize=9)
ax_p.set_ylim(0, 1000)
ax_p.invert_xaxis()

plt.suptitle('CanSat measurements vs ISA reference  (6 Feb 2026, Dübendorf)', fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig6_isa_comparison.png", dpi=130, bbox_inches="tight")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"Figures saved to: {OUTPUT_DIR}/")
print(f"  fig1_three_groups.png")
print(f"  fig2_vamos_drop.png")
print(f"  fig3_welch_psd.png")
print(f"  fig4_spectrogram_vamos.png")
print(f"  fig5_grasp_spectral.png")
print(f"  fig6_isa_comparison.png")
print(f"\nKey findings:")
print(f"  VAMOS drop: {h_peak:.0f} m AGL peak, {drop_duration:.0f} s duration, {descent_rate:.2f} m/s")
print(f"  Ground pressure: {p0_vamos:.2f} hPa")
print(f"  Sampling: VAMOS 1 Hz (Nyquist 0.5 Hz), GRASP 38 Hz (Nyquist 19 Hz)")
print("="*70)

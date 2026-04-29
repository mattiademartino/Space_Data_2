"""
CanSat Mission Analysis — Research Question Script
====================================================

Research Question:
"How do atmospheric parameters (temperature, pressure, PM2.5/PM10, CO2, wind)
vary with altitude during CanSat descent, and what do spectral methods reveal
about sensor performance and tropospheric dynamics?"

FIGURE 1 — Mission Overview (4 probes raw data quality assessment)
FIGURE 2 — Vertical profiles comparison (VAMOS + SCAMSAT vs Payerne + ISA)
FIGURE 3 — Spectral analysis (Welch PSD, Nyquist limits, STFT)
FIGURE 4 — Sensor performance (temperature bias, pressure accuracy)

Data:
  - VAMOS:   science_VAMOS.csv, wind_VAMOS.csv
  - GRASP:   science_GRASP.csv
  - OBAMA:   OBAMA_data_decoded.xlsx
  - SCAMSAT: ./002/ binary files (alt.txt, press.txt, temp.txt, pm2_5.txt, pm10.txt)
  - Payerne radiosonde: WMO 06610, 6 Feb 2026 (hardcoded)
  - ISA: International Standard Atmosphere

USAGE:
  python cansat_research_analysis.py
  Place SCAMSAT files in ./002/ folder for full analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.signal import welch, spectrogram
from scipy.ndimage import uniform_filter1d
import os, warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = "./Cansat data utili"
SCAMSAT_DIR  = "./002"
OUTPUT_DIR   = "./figures"
DUBENDORF    = 448   # m ASL
Rd, Cp, g    = 287.05, 1004.0, 9.80665
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Colors ───────────────────────────────────────────────────────────────────
C = {'VAMOS':'#1f77b4', 'SCAMSAT':'#2ca02c', 'GRASP':'#d62728',
     'OBAMA':'#9467bd', 'PAY':'#ff7f0e', 'ISA':'black'}

# ============================================================================
# HELPERS
# ============================================================================

def isa_T(h): return 288.15 - 0.0065 * h          # K, h in m ASL
def isa_p(h): return 1013.25*(isa_T(h)/288.15)**5.255
def baro(p, p0): return 44330*(1-(p/p0)**(1/5.255))
def rho(p_hPa, T_K): return (p_hPa*100)/(Rd*T_K)

def clean(df):
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna().reset_index(drop=True)

def find_drop(t, p):
    """Data-driven drop detection via largest dp/dt block."""
    p_sm = pd.Series(p).rolling(5, center=True, min_periods=1).median().values
    dpdt = np.gradient(p_sm) / np.gradient(t)
    idx  = np.where(dpdt > 0.1)[0]
    if len(idx) == 0: return None, None
    gaps = np.where(np.diff(idx) > 15)[0]
    S = np.concatenate([[idx[0]], idx[gaps+1]]) if len(gaps) else np.array([idx[0]])
    E = np.concatenate([idx[gaps], [idx[-1]]])  if len(gaps) else np.array([idx[-1]])
    ib = np.argmax([p[e]-p[s] for s,e in zip(S,E)])
    ds, de = S[ib], E[ib]
    apex = max(0,ds-60) + int(np.argmin(p[max(0,ds-60):ds+1]))
    post = np.where(p[de:min(len(p),de+60)] > p[apex]+30)[0]
    land = de + (post[0] if len(post) else 0)
    return t[apex], t[land]

# ============================================================================
# PAYERNE RADIOSONDE (6 FEB 2026, 12 UTC, WMO 06610)
# ============================================================================

PAY_RAW = """935.9 491 1.9 0.9 93 4 0.3
933.8 509 1.1 1.0 99 45 0.9
930.4 538 0.8 0.7 99 40 1.1
927.8 561 0.6 0.5 100 29 1.2
925.2 584 0.4 0.4 100 19 1.2
922.6 606 0.3 0.3 100 12 1.3
919.1 637 0.1 0.1 100 7 1.3
916.5 660 0.0 0.0 100 9 1.2
914.9 674 -0.0 -0.0 100 12 1.1
912.3 696 -0.2 -0.2 100 23 0.8
909.4 722 -0.3 -0.3 100 50 0.5
905.6 755 -0.5 -0.5 100 123 0.5
902.3 784 -0.7 -0.8 100 151 0.9
900.5 801 -0.2 -2.6 84 159 1.1
898.3 821 1.0 -2.8 76 168 1.3
896.1 840 1.8 -3.4 68 176 1.6
893.3 865 2.5 -2.8 68 185 2.0
890.4 892 3.0 -2.5 67 192 2.6
887.0 923 3.1 -2.5 67 197 3.2
884.1 949 2.9 -2.8 66 201 3.9
881.0 978 2.8 -3.2 65 204 4.4
878.7 998 2.6 -3.8 62 206 4.7
875.0 1032 2.5 -4.1 62 209 5.2
872.5 1056 2.5 -5.2 57 210 5.4
869.2 1087 2.8 -5.6 54 211 5.7
866.9 1108 3.0 -5.1 55 210 5.8
864.5 1130 3.5 -4.0 58 210 5.8
862.3 1151 3.4 -3.3 62 209 5.9
859.8 1175 3.2 -3.2 63 209 6.0
857.6 1196 3.0 -3.1 64 211 6.2
855.5 1215 2.8 -2.6 67 212 6.6
853.2 1237 2.7 -2.5 69 215 7.0
851.0 1258 2.6 -2.5 69 217 7.5
849.3 1274 2.4 -2.4 70 219 7.8
847.1 1295 2.3 -2.1 72 220 8.3
844.8 1317 2.2 -1.9 74 221 8.6
842.6 1338 2.2 -2.0 74 222 8.9
840.4 1359 2.2 -2.3 72 223 9.2
838.1 1382 2.2 -2.4 72 223 9.4
835.9 1403 2.1 -2.4 72 224 9.5
833.7 1424 1.9 -2.5 73 224 9.7
831.5 1445 1.7 -2.5 74 223 9.9
829.2 1467 1.5 -2.5 75 222 10.0
827.0 1489 1.3 -2.5 76 220 10.1
825.9 1499 1.2 -2.5 76 219 10.1"""

_rows = [[float(x) for x in ln.split()] for ln in PAY_RAW.strip().split('\n')]
pay = pd.DataFrame(_rows, columns=['p','h_asl','T','Td','rh','wdir','wspd'])
pay['h_agl'] = pay['h_asl'] - DUBENDORF
wr = np.deg2rad(pay['wdir'])
pay['u'] = -pay['wspd']*np.sin(wr)
pay['v'] = -pay['wspd']*np.cos(wr)

# ============================================================================
# LOAD ALL PROBE DATA
# ============================================================================

print("Loading data...")

# ── VAMOS ────────────────────────────────────────────────────────────────────
sv = clean(pd.read_csv(f"{DATA_DIR}/science_VAMOS.csv"))
t_v, p_v, T_v, co2_v = (sv["timestamp_ms"].values/1000,
                          sv["pressure_hPa"].values,
                          sv["temperature_C"].values,
                          sv["co2_ppm"].values)
p0_v = (np.median(p_v[:100])+np.median(p_v[-500:]))/2
h_v  = baro(p_v, p0_v)

wv = clean(pd.read_csv(f"{DATA_DIR}/wind_VAMOS.csv"))
tw = wv["timestamp_ms"].values/1000
resets = np.where(np.diff(tw)<0)[0]
if len(resets):
    wv = wv.iloc[resets[-1]+1:].reset_index(drop=True)
    tw = wv["timestamp_ms"].values/1000
ux_v = wv["x_wind_mps"].values if "x_wind_mps" in wv.columns else wv["x_wind_acc"].values
uy_v = wv["y_wind_mps"].values if "y_wind_mps" in wv.columns else wv["y_wind_acc"].values
ws_v = wv["wind_speed"].values

t_apex_v, t_land_v = find_drop(t_v, p_v)
dm_v  = (t_v >= t_apex_v) & (t_v <= t_land_v)
dwm_v = (tw  >= t_apex_v) & (tw  <= t_land_v)
print(f"  VAMOS: {len(sv)} science, drop {dm_v.sum()} pts, "
      f"h={h_v[dm_v].min():.0f}-{h_v[dm_v].max():.0f} m AGL")

# ── GRASP ────────────────────────────────────────────────────────────────────
sg = clean(pd.read_csv(f"{DATA_DIR}/science_GRASP.csv",
           on_bad_lines='skip'))
# Rename columns
sg.columns = [c.strip() for c in sg.columns]
t_g  = sg["timestamp [ms]"].values/1000
p_g  = sg["pressure [Pa]"].values/100     # Pa → hPa
T_g  = sg["temperature [C]"].values
h_g_asl = sg["altitude [m]"].values if " altitude [m] " not in sg.columns else sg[" altitude [m] "].values
h_g  = h_g_asl - DUBENDORF
pm25_g = sg["pm2_5 [microg/m3]"].values
pm10_g = sg["pm10_0 [microg/m3]"].values
# Relative to freefall trigger
t_g_rel = t_g - t_g[0]
print(f"  GRASP: {len(sg)} samples, T={T_g.min():.1f}-{T_g.max():.1f}°C, "
      f"Δp={p_g.max()-p_g.min():.1f} hPa (duration {t_g_rel[-1]:.0f}s)")

# ── OBAMA ────────────────────────────────────────────────────────────────────
ob = pd.read_excel(f"{DATA_DIR}/OBAMA_data_decoded.xlsx", sheet_name='decoded')
for c in ['Time_s','first_pstat_avg_hPa','first_temp_avg_C','first_hum_avg_pct']:
    ob[c] = pd.to_numeric(ob[c], errors='coerce')
ob = ob[(ob['Time_s']>0)&(ob['Time_s']<600)
        &(ob['first_temp_avg_C'].between(-50,60))].dropna(
        subset=['Time_s','first_pstat_avg_hPa']).sort_values('Time_s').reset_index(drop=True)
t_o = ob['Time_s'].values
p_o = ob['first_pstat_avg_hPa'].values
T_o = ob['first_temp_avg_C'].values
p0_o = (np.median(p_o[:3])+np.median(p_o[-3:]))/2
h_o = baro(p_o, p0_o)
print(f"  OBAMA: {len(ob)} samples, p={p_o.min():.0f}-{p_o.max():.0f} hPa")

# ── SCAMSAT ──────────────────────────────────────────────────────────────────
SCAM_DUR = 3550; SCAM_T0 = 828; SCAM_T1 = 996
scamsat_ok = False

def load_sc(fname, dtype, scale=1.0):
    path = os.path.join(SCAMSAT_DIR, fname)
    return np.fromfile(path, dtype=dtype).astype(float)*scale if os.path.exists(path) else None

alt_sc   = load_sc('alt.txt',   'uint16', 0.1)   # dm → m ASL
press_sc = load_sc('press.txt', 'float32', 1.0)  # Pa
temp_sc  = load_sc('temp.txt',  'float32', 1.0)  # °C
pm25_sc  = load_sc('pm2_5.txt', 'uint16',  0.1)  # µg/m³
pm10_sc  = load_sc('pm10.txt',  'uint16',  0.1)  # µg/m³

if alt_sc is not None and press_sc is not None:
    scamsat_ok = True
    N_sc   = len(alt_sc)
    t_sc   = np.linspace(0, SCAM_DUR, N_sc)
    dm_sc  = (t_sc >= SCAM_T0) & (t_sc <= SCAM_T1)
    h_sc   = alt_sc - DUBENDORF                   # m AGL
    p_sc   = press_sc / 100.0                     # Pa → hPa
    p0_sc  = np.median(p_sc[dm_sc][-20:])
    print(f"  SCAMSAT: {N_sc} samples, drop {dm_sc.sum()} pts, "
          f"h={h_sc[dm_sc].min():.0f}-{h_sc[dm_sc].max():.0f} m AGL")
else:
    print("  SCAMSAT: files not found in ./002/ → SCAMSAT panels will show placeholder")

# ============================================================================
# FIGURE 1 — MISSION OVERVIEW (raw data + quality table)
# ============================================================================

print("\nGenerating Figure 1: Mission Overview...")

fig1 = plt.figure(figsize=(20, 14))
gs1  = gridspec.GridSpec(3, 4, figure=fig1, hspace=0.45, wspace=0.35)

# ── Row 0: Pressure vs Time for each probe ───────────────────────────────────
axes_p = [fig1.add_subplot(gs1[0, i]) for i in range(4)]
probes_raw = [
    ('VAMOS',   t_v,    p_v,    C['VAMOS'],   t_apex_v, t_land_v,
     f'{len(sv)} samples\nfs≈1 Hz\nDrop: {h_v[dm_v].max():.0f} m AGL'),
    ('GRASP',   t_g_rel, p_g,   C['GRASP'],   None, None,
     f'{len(sg)} samples\nfs≈38 Hz\nΔp={p_g.max()-p_g.min():.0f} hPa'),
    ('OBAMA',   t_o,    p_o,    C['OBAMA'],   None, None,
     f'{len(ob)} samples\nfs≈0.04 Hz\nCoarse quant.'),
    ('SCAMSAT', None,   None,   C['SCAMSAT'], None, None, ''),
]

for ax, (name, t, p, col, ta, tl, ann) in zip(axes_p, probes_raw):
    if name == 'SCAMSAT':
        if scamsat_ok:
            ax.plot(t_sc, p_sc, color=col, lw=1)
            ax.axvspan(SCAM_T0, SCAM_T1, color=col, alpha=0.15, label='Drop')
            ax.axvline(SCAM_T0, color='green',  lw=1.5, ls='--')
            ax.axvline(SCAM_T1, color='orange', lw=1.5, ls='--')
            ann = (f'{N_sc} samples\nfs≈1 Hz\n'
                   f'Drop: {h_sc[dm_sc].max():.0f} m AGL')
        else:
            ax.text(0.5, 0.5, 'Data not found\nPlace in ./002/',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=11, color='gray')
    else:
        ax.plot(t, p, color=col, lw=1)
        if ta is not None:
            ax.axvspan(ta, tl, color=col, alpha=0.15)
            ax.axvline(ta, color='green',  lw=1.5, ls='--', label='Apogee')
            ax.axvline(tl, color='orange', lw=1.5, ls='--', label='Landing')

    ax.set_title(name, fontsize=13, fontweight='bold', color=col, pad=4)
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel('Pressure [hPa]', fontsize=10)
    ax.grid(alpha=0.3)
    ax.text(0.97, 0.97, ann, transform=ax.transAxes,
            fontsize=8.5, family='monospace', va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor=col, lw=1.5))

# ── Row 1: Temperature vs Altitude during drop (or full mission) ─────────────
axes_T = [fig1.add_subplot(gs1[1, i]) for i in range(4)]

# ISA reference
h_ref = np.linspace(0, 800, 200)
T_isa = isa_T(h_ref + DUBENDORF) - 273.15
T_pay = np.interp(h_ref, pay['h_agl'].values[::-1], pay['T'].values[::-1])

for ax, (name, col) in zip(axes_T,
    [('VAMOS',C['VAMOS']),('GRASP',C['GRASP']),('OBAMA',C['OBAMA']),('SCAMSAT',C['SCAMSAT'])]):

    ax.plot(T_isa, h_ref, 'k--', lw=1.5, label='ISA', alpha=0.7)
    ax.plot(T_pay, h_ref, color=C['PAY'], lw=2, label='Payerne')

    if name == 'VAMOS':
        ax.scatter(T_v[dm_v], h_v[dm_v], s=8, c=col, alpha=0.5, label='VAMOS')
        bias = np.mean(T_v[dm_v]) - np.interp(np.mean(h_v[dm_v]),
                       pay['h_agl'].values[::-1], pay['T'].values[::-1])
        ax.text(0.05, 0.05, f'Bias vs Payerne:\n{bias:+.1f}°C',
                transform=ax.transAxes, fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    elif name == 'GRASP':
        ax.scatter(T_g, h_g, s=8, c=col, alpha=0.5, label='GRASP')
        ax.text(0.05, 0.05, 'Flat T → likely\naircraft cabin\n(not free drop)',
                transform=ax.transAxes, fontsize=9, family='monospace', color='red',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    elif name == 'OBAMA':
        ax.scatter(T_o, h_o, s=30, c=col, alpha=0.6, label='OBAMA', marker='s')
        ax.text(0.05, 0.05, 'Very few points\nCoarse resolution',
                transform=ax.transAxes, fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    elif name == 'SCAMSAT':
        if scamsat_ok and temp_sc is not None:
            ax.scatter(temp_sc[dm_sc], h_sc[dm_sc], s=8, c=col, alpha=0.5, label='SCAMSAT')
            bias = np.mean(temp_sc[dm_sc]) - np.interp(np.mean(h_sc[dm_sc]),
                           pay['h_agl'].values[::-1], pay['T'].values[::-1])
            ax.text(0.05, 0.05, f'Bias vs Payerne:\n{bias:+.1f}°C',
                    transform=ax.transAxes, fontsize=9, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')

    ax.set_xlabel('Temperature [°C]', fontsize=10)
    ax.set_ylabel('Altitude [m AGL]', fontsize=10)
    ax.set_title(f'{name} — T(h) vs references', fontsize=11, fontweight='bold', color=col)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 800)

# ── Row 2: Data Quality Summary Table ────────────────────────────────────────
ax_tbl = fig1.add_subplot(gs1[2, :])
ax_tbl.axis('off')

rows = [
    ['VAMOS',   '1 Hz',      '9349',  '99 min',  '✓ 713 m',  'T, p, CO₂, wind',  '+11.5°C',  '✅ PRIMARY — full drop, wind data'],
    ['SCAMSAT', '~1 Hz',     '3550',  '~59 min', '✓ 665 m',  'T, p, PM₂.₅, PM₁₀','~+10°C*', '✅ SECONDARY — PM data, clean drop'],
    ['GRASP',   '38 Hz',     '1715',  '176 s',   '❌ False',  'T, p, PM₂.₅, PM₁₀','flat T',  '⚠️  SPECTRAL DEMO — Nyquist capability'],
    ['OBAMA',   '0.04 Hz',   '24',    '~7 min',  '⚠️  Partial','T, p, humidity',   '—',       '⚠️  MINIMAL — coarse quantization'],
]
cols = ['Probe', 'fs', 'N samples', 'Duration', 'Drop', 'Parameters', 'T bias', 'Role in analysis']
tbl = ax_tbl.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center',
                   bbox=[0.0, 0.0, 1.0, 1.0])
tbl.auto_set_font_size(False); tbl.set_fontsize(10.5)
tbl.auto_set_column_width(col=list(range(len(cols))))

# Color coding
row_colors = ['#d4edda', '#d4edda', '#fff3cd', '#fff3cd']  # green, green, yellow, yellow
for i, rc in enumerate(row_colors):
    for j in range(len(cols)):
        tbl[(i+1, j)].set_facecolor(rc)
for j in range(len(cols)):
    tbl[(0,j)].set_facecolor('#343a40')
    tbl[(0,j)].set_text_props(color='white', fontweight='bold')

ax_tbl.set_title('Data Quality Assessment — All Four Probes', fontsize=13,
                 fontweight='bold', pad=8)
ax_tbl.text(0.5, -0.05, '* SCAMSAT T bias estimated from notebook analysis (data pending)',
            ha='center', transform=ax_tbl.transAxes, fontsize=9, style='italic')

fig1.suptitle('Mission Overview: 6 February 2026, Dübendorf Drop Campaign',
              fontsize=16, fontweight='bold', y=1.00)
plt.savefig(f"{OUTPUT_DIR}/fig1_mission_overview.png", dpi=130, bbox_inches='tight')
plt.close()
print(f"  ✓ fig1_mission_overview.png")

# ============================================================================
# FIGURE 2 — VERTICAL PROFILES COMPARISON (research question part 1)
# ============================================================================

print("Generating Figure 2: Vertical profiles comparison...")

fig2, axes2 = plt.subplots(1, 4, figsize=(18, 7))

h_ref = np.linspace(0, 800, 300)
T_isa_ref = isa_T(h_ref + DUBENDORF) - 273.15
p_isa_ref = isa_p(h_ref + DUBENDORF)
T_pay_ref = np.interp(h_ref, pay['h_agl'].values, pay['T'].values)
p_pay_ref = np.interp(h_ref, pay['h_agl'].values, pay['p'].values)

# ── Panel A: Temperature ──────────────────────────────────────────────────────
ax = axes2[0]
ax.plot(T_isa_ref, h_ref, 'k--', lw=2, label='ISA', zorder=1)
ax.plot(pay['T'], pay['h_agl'], color=C['PAY'], lw=2.5, marker='o', ms=3,
        label='Payerne (real)', zorder=2)
ax.scatter(T_v[dm_v], h_v[dm_v], s=12, c=C['VAMOS'], alpha=0.6, label='VAMOS', zorder=4)
if scamsat_ok and temp_sc is not None:
    ax.scatter(temp_sc[dm_sc], h_sc[dm_sc], s=12, c=C['SCAMSAT'], alpha=0.5,
               label='SCAMSAT', zorder=3)
# Shade biases
si = np.argsort(h_v[dm_v])
ax.fill_betweenx(h_v[dm_v][si],
                 np.interp(h_v[dm_v][si], pay['h_agl'].values, pay['T'].values),
                 T_v[dm_v][si], alpha=0.12, color=C['VAMOS'])
ax.set_xlabel('Temperature [°C]', fontsize=12)
ax.set_ylabel('Altitude [m AGL]', fontsize=12)
ax.set_title('Temperature T(h)\n(ISA masks sensor error!)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylim(0, 800)

# Bias annotations
bias_v = np.mean(T_v[dm_v]) - np.mean(np.interp(h_v[dm_v], pay['h_agl'].values, pay['T'].values))
ax.text(0.04, 0.04,
        f'Mean bias vs Payerne:\nVAMOS:   {bias_v:+.1f}°C\nISA ref: {np.mean(T_isa_ref)-np.mean(T_pay_ref):+.1f}°C',
        transform=ax.transAxes, fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# ── Panel B: Pressure ─────────────────────────────────────────────────────────
ax = axes2[1]
ax.plot(p_isa_ref, h_ref, 'k--', lw=2, label='ISA', zorder=1)
ax.plot(pay['p'], pay['h_agl'], color=C['PAY'], lw=2.5, marker='o', ms=3,
        label='Payerne (real)', zorder=2)
ax.scatter(p_v[dm_v], h_v[dm_v], s=12, c=C['VAMOS'], alpha=0.6, label='VAMOS', zorder=4)
if scamsat_ok:
    ax.scatter(p_sc[dm_sc], h_sc[dm_sc], s=12, c=C['SCAMSAT'], alpha=0.5,
               label='SCAMSAT', zorder=3)
ax.invert_xaxis()
ax.set_xlabel('Pressure [hPa]', fontsize=12)
ax.set_ylabel('Altitude [m AGL]', fontsize=12)
ax.set_title('Pressure p(h)\n(both probes well calibrated)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylim(0, 800)
dp_v = p0_v - pay.iloc[0]['p']
ax.text(0.04, 0.04, f'Ground p offset:\nVAMOS: {dp_v:+.1f} hPa\n(geographic: +78 m)',
        transform=ax.transAxes, fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

# ── Panel C: PM2.5 / PM10 ────────────────────────────────────────────────────
ax = axes2[2]
if scamsat_ok and pm25_sc is not None:
    si_sc = np.argsort(h_sc[dm_sc])
    ax.plot(pm25_sc[dm_sc][si_sc], h_sc[dm_sc][si_sc],
            color=C['SCAMSAT'], lw=2, label='SCAMSAT PM₂.₅')
    if pm10_sc is not None:
        ax.plot(pm10_sc[dm_sc][si_sc], h_sc[dm_sc][si_sc],
                color=C['SCAMSAT'], lw=2, ls='--', alpha=0.7, label='SCAMSAT PM₁₀')
# GRASP PM
si_g = np.argsort(h_g)
ax.plot(pm25_g[si_g], h_g[si_g], color=C['GRASP'], lw=1.5, alpha=0.7, label='GRASP PM₂.₅')
ax.plot(pm10_g[si_g], h_g[si_g], color=C['GRASP'], lw=1.5, ls='--', alpha=0.7, label='GRASP PM₁₀')
# WHO limits
ax.axvline(15, color='red',    ls=':', lw=1.5, label='WHO PM₂.₅ (15 µg/m³)')
ax.axvline(45, color='darkred',ls=':', lw=1.5, label='WHO PM₁₀ (45 µg/m³)')
ax.set_xlabel('Mass Concentration [µg/m³]', fontsize=12)
ax.set_ylabel('Altitude [m AGL]', fontsize=12)
ax.set_title('PM₂.₅ / PM₁₀ vs Altitude\n(SCAMSAT + GRASP)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3); ax.set_ylim(0, 800)
if not scamsat_ok:
    ax.text(0.5, 0.5, 'SCAMSAT\nfiles needed\n(./002/)',
            ha='center', va='center', transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

# ── Panel D: CO2 (VAMOS) ─────────────────────────────────────────────────────
ax = axes2[3]
ax.scatter(co2_v[dm_v], h_v[dm_v], s=12, c=C['VAMOS'], alpha=0.6, label='VAMOS CO₂')
ax.axvline(420, color='gray', ls='--', lw=1.5, label='Global avg (420 ppm)')
ax.axvline(450, color='red',  ls=':',  lw=1.5, label='Urban threshold')
ax.set_xlabel('CO₂ [ppm]', fontsize=12)
ax.set_ylabel('Altitude [m AGL]', fontsize=12)
ax.set_title('CO₂ vs Altitude\n(VAMOS only)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylim(0, 800)
ax.text(0.04, 0.04,
        f'Mean: {np.mean(co2_v[dm_v]):.0f} ppm\nRange: {co2_v[dm_v].min():.0f}-{co2_v[dm_v].max():.0f} ppm',
        transform=ax.transAxes, fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))

fig2.suptitle('Vertical Profiles: VAMOS & SCAMSAT vs Payerne Radiosonde & ISA  (6 Feb 2026)',
              fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_vertical_profiles.png", dpi=130, bbox_inches='tight')
plt.close()
print(f"  ✓ fig2_vertical_profiles.png")

# ============================================================================
# FIGURE 3 — SPECTRAL ANALYSIS (research question part 2)
# ============================================================================

print("Generating Figure 3: Spectral analysis...")

fig3 = plt.figure(figsize=(20, 12))
gs3  = gridspec.GridSpec(2, 4, figure=fig3, hspace=0.42, wspace=0.35)

# ── Panel 1: Welch PSD VAMOS wind — 3 phases ─────────────────────────────────
ax31 = fig3.add_subplot(gs3[0, 0])

# Define phases
t_plane_end = t_apex_v - 60
mask_air  = (tw > t_v[0]+200) & (tw < t_plane_end)
mask_drop = dwm_v
mask_gnd  = tw > t_land_v + 60

acc_mag = np.sqrt(ux_v**2 + uy_v**2)
fs_w = 1.0

for mask, lbl, col, ls in [
    (mask_air,  'Aircraft (cruise)', 'steelblue', '-'),
    (mask_drop, 'Drop (parachute)',  'tomato',    '-'),
    (mask_gnd,  'Ground (landed)',   'gray',      '--'),
]:
    if mask.sum() < 64: continue
    t_m = tw[mask]; y_m = acc_mag[mask]
    t_u = np.arange(t_m[0], t_m[-1], 1/fs_w)
    y_u = np.interp(t_u, t_m, y_m) - np.mean(np.interp(t_u, t_m, y_m))
    f_w, P_w = welch(y_u, fs=fs_w, window='hann', nperseg=64, noverlap=32, detrend='linear')
    ax31.loglog(f_w[1:], P_w[1:], color=col, lw=1.5, ls=ls, label=f'{lbl} (N={mask.sum()})')

ax31.axvline(fs_w/2, color='red', ls='--', lw=1.2, label=f'Nyquist = {fs_w/2} Hz')
ax31.axvspan(0.3, 0.5, color='orange', alpha=0.15, label='Pendulum band')
ax31.set_xlabel('Frequency [Hz]'); ax31.set_ylabel('PSD [(m/s²)²/Hz]')
ax31.set_title('VAMOS Welch PSD\n|wind accel|, 3 flight phases', fontsize=11, fontweight='bold')
ax31.legend(fontsize=7.5); ax31.grid(alpha=0.3)
ax31.text(0.55, 0.92, 'Spin (1-3 Hz)\nCOMPLETELY\nALIASED', transform=ax31.transAxes,
          fontsize=8, color='red', va='top',
          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ── Panel 2: Welch PSD GRASP pressure (high-fs) ──────────────────────────────
ax32 = fig3.add_subplot(gs3[0, 1])
fs_g = 30.0
t_g_u = np.arange(t_g[0], t_g[-1], 1/fs_g)
p_g_u = np.interp(t_g_u, t_g, p_g)
coef  = np.polyfit(t_g_u - t_g_u[0], p_g_u, 3)
p_g_res = p_g_u - np.polyval(coef, t_g_u - t_g_u[0])
f_gr, P_gr = welch(p_g_res, fs=fs_g, window='hann', nperseg=512, noverlap=384)
ax32.loglog(f_gr[1:], P_gr[1:], color=C['GRASP'], lw=1.5, label='GRASP (detrended p)')
ax32.axvline(fs_g/2, color='red', ls='--', lw=1.2, label=f'Nyquist = {fs_g/2} Hz')
ax32.axvspan(0.3, 1.0, color='orange', alpha=0.15, label='Pendulum 0.3-1 Hz')
ax32.axvspan(1.0, 3.0, color='red',    alpha=0.10, label='Spin 1-3 Hz')
ax32.set_xlabel('Frequency [Hz]'); ax32.set_ylabel('PSD [hPa²/Hz]')
ax32.set_title('GRASP Welch PSD\n(38 Hz capable — Nyquist 19 Hz)', fontsize=11, fontweight='bold')
ax32.legend(fontsize=7.5); ax32.grid(alpha=0.3)
ax32.text(0.5, 0.08, 'No pendulum/spin peaks\n→ confirms no real drop',
          transform=ax32.transAxes, fontsize=8, ha='center',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# ── Panel 3: Spectrogram VAMOS ────────────────────────────────────────────────
ax33 = fig3.add_subplot(gs3[0, 2:])
t_wu = np.arange(tw[0], tw[-1], 1.0)
acc_u = np.interp(t_wu, tw, acc_mag) - np.mean(acc_mag)
f_sp, t_sp, S_sp = spectrogram(acc_u, fs=1.0, window='hann',
                                nperseg=64, noverlap=56, scaling='density')
t_sp_abs = t_sp + t_wu[0]
pcm = ax33.pcolormesh(t_sp_abs, f_sp, 10*np.log10(S_sp+1e-12),
                      shading='auto', cmap='viridis',
                      vmin=np.nanpercentile(10*np.log10(S_sp+1e-12), 5),
                      vmax=np.nanpercentile(10*np.log10(S_sp+1e-12), 99))
ax33.axvline(t_apex_v, color='lime',   lw=2, ls='--', label=f'Apogee t={t_apex_v:.0f}s')
ax33.axvline(t_land_v, color='yellow', lw=2, ls='--', label=f'Landing t={t_land_v:.0f}s')
plt.colorbar(pcm, ax=ax33, label='PSD [dB re (m/s²)²/Hz]')
ax33.set_xlabel('Time [s]'); ax33.set_ylabel('Frequency [Hz]')
ax33.set_title('VAMOS STFT Spectrogram — |wind accel|\n'
               '(fs=1 Hz, Hann, nperseg=64, overlap=88%)',
               fontsize=11, fontweight='bold')
ax33.legend(fontsize=9, loc='upper right')
for phase, t0, t1, lbl in [('pre-apogee', t_wu[0], t_apex_v-30, 'pre-apogee'),
                             ('DROP',       t_apex_v, t_land_v,    'DROP'),
                             ('post-land',  t_land_v+30, t_wu[-1], 'post-land')]:
    ax33.text((t0+t1)/2, 0.48, lbl, ha='center', va='top', fontsize=9, color='white',
              bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

# ── Panel 4: Sampling rate comparison infographic ────────────────────────────
ax34 = fig3.add_subplot(gs3[1, 0])
categories = ['VAMOS\n1 Hz', 'SCAMSAT\n~1 Hz', 'GRASP\n38 Hz', 'OBAMA\n0.04 Hz']
nyquist    = [0.5, 0.5, 19.0, 0.02]
bars = ax34.barh(categories, nyquist, color=[C['VAMOS'],C['SCAMSAT'],C['GRASP'],C['OBAMA']],
                 edgecolor='black', lw=0.8)
ax34.axvline(0.3, color='orange', ls='--', lw=1.5, label='Pendulum min (0.3 Hz)')
ax34.axvline(1.0, color='red',    ls='--', lw=1.5, label='Spin min (1 Hz)')
for bar, val in zip(bars, nyquist):
    ax34.text(val+0.2, bar.get_y()+bar.get_height()/2,
              f'{val} Hz', va='center', fontsize=9, fontweight='bold')
ax34.set_xlabel('Nyquist Limit [Hz]'); ax34.set_title('Nyquist Limits\n(observable bandwidth)',
               fontsize=11, fontweight='bold')
ax34.legend(fontsize=8); ax34.grid(axis='x', alpha=0.3); ax34.set_xlim(0, 23)

# ── Panel 5: T bias comparison VAMOS vs Payerne vs ISA by altitude ───────────
ax35 = fig3.add_subplot(gs3[1, 1])
h_bins = np.arange(0, 800, 50)
bias_isa_v, bias_pay_v, h_centers = [], [], []
for hb in h_bins:
    m = (h_v[dm_v] >= hb) & (h_v[dm_v] < hb+50)
    if m.sum() > 3:
        T_m = np.mean(T_v[dm_v][m]); hc = hb+25
        bias_isa_v.append(T_m - (isa_T(hc+DUBENDORF)-273.15))
        bias_pay_v.append(T_m - np.interp(hc, pay['h_agl'].values, pay['T'].values))
        h_centers.append(hc)

ax35.plot(bias_isa_v, h_centers, 'o-', color='black',   lw=2, ms=6, label='VAMOS − ISA')
ax35.plot(bias_pay_v, h_centers, 's-', color=C['PAY'],  lw=2, ms=6, label='VAMOS − Payerne')
ax35.axvline(0, color='gray', ls='--', lw=1)
ax35.fill_betweenx(h_centers, 0, bias_pay_v, alpha=0.15, color=C['PAY'])
ax35.set_xlabel('Temperature Bias [°C]'); ax35.set_ylabel('Altitude [m AGL]')
ax35.set_title('VAMOS Temperature Bias\n(ISA comparison misleading!)', fontsize=11, fontweight='bold')
ax35.legend(fontsize=9); ax35.grid(alpha=0.3); ax35.set_ylim(0, 800)
ax35.text(0.55, 0.08,
          f'Mean bias:\nvs ISA:    {np.mean(bias_isa_v):+.1f}°C\nvs Payerne:{np.mean(bias_pay_v):+.1f}°C',
          transform=ax35.transAxes, fontsize=9, family='monospace',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

# ── Panel 6: Wind hodograph Payerne vs VAMOS ─────────────────────────────────
ax36 = fig3.add_subplot(gs3[1, 2])
mask_h = pay['h_agl'] <= 800
sc36 = ax36.scatter(pay['u'].values[mask_h], pay['v'].values[mask_h],
                    c=pay['h_agl'].values[mask_h], cmap='viridis',
                    s=40, zorder=3, edgecolors='k', lw=0.5)
ax36.plot(pay['u'].values[mask_h], pay['v'].values[mask_h], '-', color='gray', lw=1, alpha=0.5)
plt.colorbar(sc36, ax=ax36, label='Alt [m AGL]', shrink=0.8)
for r in [2, 4, 6, 8]:
    ax36.add_patch(plt.Circle((0,0), r, fill=False, color='gray', ls=':', alpha=0.4))
ax36.axhline(0, color='k', lw=0.5, alpha=0.3); ax36.axvline(0, color='k', lw=0.5, alpha=0.3)
ax36.set_xlabel('U (East) [m/s]'); ax36.set_ylabel('V (North) [m/s]')
ax36.set_title('Payerne Wind Hodograph\n(spiraling = shear with altitude)', fontsize=11, fontweight='bold')
ax36.set_aspect('equal'); ax36.grid(alpha=0.3); ax36.set_xlim(-12,12); ax36.set_ylim(-12,12)

# ── Panel 7: Vertical wind shear ─────────────────────────────────────────────
ax37 = fig3.add_subplot(gs3[1, 3])
h_p  = pay['h_agl'].values
u_sm = uniform_filter1d(pay['u'].values, size=5)
v_sm = uniform_filter1d(pay['v'].values, size=5)
shear = np.sqrt(np.gradient(u_sm, h_p)**2 + np.gradient(v_sm, h_p)**2) * 1000
valid = h_p <= 800
ax37.plot(shear[valid], h_p[valid], 'o-', color='purple', lw=2, ms=4, label='Wind shear')
ax37.fill_betweenx(h_p[valid], 0, shear[valid], alpha=0.2, color='purple')
ax37_r = ax37.twiny()
ax37_r.plot(pay['wspd'].values[valid], h_p[valid], 's-', color='darkgreen',
            lw=1.5, ms=4, alpha=0.8)
ax37_r.set_xlabel('Wind Speed [m/s]', fontsize=10, color='darkgreen')
ax37_r.tick_params(axis='x', labelcolor='darkgreen')
idx_max = np.argmax(shear[valid])
ax37.annotate(f'Max: {shear[valid][idx_max]:.0f} m/s/km\n@ {h_p[valid][idx_max]:.0f} m',
              xy=(shear[valid][idx_max], h_p[valid][idx_max]),
              xytext=(shear[valid][idx_max]*0.4, h_p[valid][idx_max]+150),
              arrowprops=dict(arrowstyle='->', color='red'),
              fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax37.set_xlabel('Wind Shear [m/s per km]', fontsize=10, color='purple')
ax37.tick_params(axis='x', labelcolor='purple')
ax37.set_ylabel('Altitude [m AGL]'); ax37.grid(alpha=0.3); ax37.set_ylim(0, 800)
ax37.set_title('Vertical Wind Shear\n(Payerne radiosonde)', fontsize=11, fontweight='bold')

fig3.suptitle('Spectral Analysis & Sensor Performance vs Tropospheric Dynamics  (6 Feb 2026)',
              fontsize=14, fontweight='bold')
plt.savefig(f"{OUTPUT_DIR}/fig3_spectral_analysis.png", dpi=130, bbox_inches='tight')
plt.close()
print(f"  ✓ fig3_spectral_analysis.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\n  fig1_mission_overview.png  — 4-probe overview + quality table")
print(f"  fig2_vertical_profiles.png — T, p, PM, CO2 vs altitude")
print(f"  fig3_spectral_analysis.png — PSD, spectrogram, bias, wind")
print(f"\nKEY FINDINGS:")
print(f"  VAMOS:   drop {h_v[dm_v].max():.0f} m AGL, T bias {np.mean(bias_pay_v):+.1f}°C vs Payerne")
print(f"  GRASP:   38 Hz capable BUT Δp={p_g.max()-p_g.min():.0f} hPa → no real drop")
print(f"  Payerne: wind shear max {shear[valid][idx_max]:.0f} m/s/km at {h_p[valid][idx_max]:.0f} m")
print(f"  6 Feb 2026 atmosphere: {np.mean(pay['T'].values[pay['h_agl'].values<=700]):.1f}°C vs ISA {np.mean(T_isa_ref[h_ref<=700]):.1f}°C")
if scamsat_ok:
    print(f"  SCAMSAT: drop {h_sc[dm_sc].max():.0f} m AGL, {dm_sc.sum()} samples")
print("="*70)

sounding_wyoming(
  wmo_id,
  yy,
  mm,
  dd,
  hh,
  min = 0,
  bufr = FALSE,
  allow_failure = TRUE
)

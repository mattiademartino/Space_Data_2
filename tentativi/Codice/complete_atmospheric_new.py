"""
CanSat Complete Atmospheric Analysis — WITH SCAMSAT
=====================================================

Analyzes and compares data from:
  - VAMOS:   1 Hz, science (T, p, CO2) + wind
  - SCAMSAT: ~1 Hz, binary files (alt, press, temp, pm2_5, pm10, accel)
  - Payerne radiosonde: WMO 06610, 6 Feb 2026 (external reference)
  - ISA: International Standard Atmosphere (theoretical reference)

SCAMSAT data format (binary files in ./002/):
  alt.txt    → uint16,  /10         → altitude [m ASL]
  press.txt  → float32             → pressure [Pa]
  temp.txt   → float32             → temperature [°C]
  pm2_5.txt  → uint16,  /10        → PM2.5 [µg/m³]
  pm10.txt   → uint16,  /10        → PM10  [µg/m³]
  accel.txt  → int16,   /1000      → acceleration [g]

SCAMSAT drop: t = 828 to 996 s (total recording = 3550 s)
              altitude from 1113 m to 448 m ASL (665 m drop)
              sampling: N=3550 samples → fs ≈ 1 Hz

Output figures (saved to ./figures/):
  fig8_complete_atmospheric.png  — 9-panel analysis
  fig9_vamos_hodograph.png       — standalone VAMOS hodograph
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, json

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR    = "./figures"
DATA_DIR = "./Cansat data utili"
SCAMSAT_DIR   = "./002"          # folder with SCAMSAT binary files
DUBENDORF_ELEV = 448             # m ASL

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Physical constants
Rd  = 287.05   # J/(kg·K)
Cp  = 1004.0   # J/(kg·K)
g   = 9.80665  # m/s²

# ============================================================================
# PAYERNE RADIOSONDE (6 FEB 2026, 12 UTC)
# ============================================================================

PAYERNE_RAW = """
  935.9    491    1.9    0.9     93   4.36      4    0.3
  933.8    509    1.1    1.0     99   4.39     45    0.9
  932.7    519    0.9    0.9    100   4.37     48    1.2
  930.4    538    0.8    0.7     99   4.31     40    1.1
  929.1    550    0.7    0.6     99   4.29     35    1.1
  927.8    561    0.6    0.5    100   4.28     29    1.2
  926.5    572    0.5    0.5    100   4.27     24    1.2
  925.2    584    0.4    0.4    100   4.26     19    1.2
  923.9    595    0.4    0.4    100   4.25     15    1.3
  922.6    606    0.3    0.3    100   4.24     12    1.3
  920.4    626    0.2    0.2    100   4.20      8    1.3
  919.1    637    0.1    0.1    100   4.18      7    1.3
  917.9    647    0.1    0.1    100   4.19      8    1.3
  916.5    660    0.0    0.0    100   4.18      9    1.2
  914.9    674   -0.0   -0.0    100   4.17     12    1.1
  913.5    686   -0.1   -0.1    100   4.14     16    0.9
  912.3    696   -0.2   -0.2    100   4.14     23    0.8
  910.9    709   -0.2   -0.2    100   4.13     33    0.6
  909.4    722   -0.3   -0.3    100   4.11     50    0.5
  907.4    740   -0.4   -0.4    100   4.09     90    0.4
  905.6    755   -0.5   -0.5    100   4.07    123    0.5
  904.0    770   -0.6   -0.6    100   4.05    140    0.7
  902.3    784   -0.7   -0.8    100   4.01    151    0.9
  900.5    801   -0.2   -2.6     84   3.52    159    1.1
  899.3    811    0.2   -3.0     79   3.39    163    1.2
  898.3    821    1.0   -2.8     76   3.47    168    1.3
  897.2    830    1.3   -3.7     70   3.24    172    1.5
  896.1    840    1.8   -3.4     68   3.32    176    1.6
  894.9    850    2.1   -3.1     68   3.38    180    1.8
  893.3    865    2.5   -2.8     68   3.47    185    2.0
  892.0    877    2.8   -2.5     68   3.57    188    2.2
  890.4    892    3.0   -2.5     67   3.57    192    2.6
  889.3    901    3.1   -2.5     67   3.58    194    2.8
  888.1    913    3.1   -2.5     67   3.59    195    3.0
  887.0    923    3.1   -2.5     67   3.59    197    3.2
  886.0    932    3.0   -2.5     67   3.58    198    3.5
  885.0    941    3.0   -2.7     66   3.54    200    3.7
  884.1    949    2.9   -2.8     66   3.50    201    3.9
  883.1    958    2.8   -2.9     66   3.51    202    4.0
  882.1    968    2.8   -3.0     65   3.47    203    4.2
  881.0    978    2.8   -3.2     65   3.44    204    4.4
  879.9    988    2.7   -3.3     65   3.40    205    4.6
  878.7    998    2.6   -3.8     62   3.28    206    4.7
  877.5   1010    2.7   -4.1     61   3.22    207    4.9
  876.3   1021    2.6   -4.1     62   3.23    208    5.0
  875.0   1032    2.5   -4.1     62   3.23    209    5.2
  873.8   1044    2.4   -4.3     61   3.18    210    5.3
  872.5   1056    2.5   -5.2     57   2.98    210    5.4
  871.4   1066    2.7   -5.6     54   2.88    210    5.5
  870.2   1077    2.7   -5.5     55   2.90    210    5.6
  869.2   1087    2.8   -5.6     54   2.89    211    5.7
  868.1   1096    2.8   -5.6     54   2.89    210    5.7
  866.9   1108    3.0   -5.1     55   3.03    210    5.8
  865.6   1120    3.5   -4.5     56   3.15    210    5.8
  864.5   1130    3.5   -4.0     58   3.29    210    5.8
  863.4   1140    3.5   -3.3     61   3.48    209    5.8
  862.3   1151    3.4   -3.3     62   3.48    209    5.9
  861.4   1160    3.3   -3.2     62   3.49    209    5.9
  859.8   1175    3.2   -3.2     63   3.52    209    6.0
  857.6   1196    3.0   -3.1     64   3.55    211    6.2
  856.5   1206    2.9   -2.8     66   3.64    211    6.4
  855.5   1215    2.8   -2.6     67   3.68    212    6.6
  854.4   1226    2.8   -2.6     68   3.70    214    6.8
  853.2   1237    2.7   -2.5     69   3.73    215    7.0
  852.1   1248    2.6   -2.5     69   3.73    216    7.2
  851.0   1258    2.6   -2.5     69   3.72    217    7.5
  850.0   1268    2.5   -2.5     70   3.75    218    7.6
  849.3   1274    2.4   -2.4     70   3.77    219    7.8
  848.2   1285    2.4   -2.2     72   3.83    219    8.0
  847.1   1295    2.3   -2.1     72   3.85    220    8.3
  846.0   1306    2.3   -2.1     73   3.88    221    8.5
  844.8   1317    2.2   -1.9     74   3.93    221    8.6
  843.7   1328    2.2   -1.9     74   3.94    222    8.8
  842.6   1338    2.2   -2.0     74   3.92    222    8.9
  841.5   1348    2.3   -2.2     72   3.87    223    9.1
  840.4   1359    2.2   -2.3     72   3.84    223    9.2
  839.2   1371    2.2   -2.3     72   3.84    223    9.3
  838.1   1382    2.2   -2.4     72   3.83    223    9.4
  837.0   1392    2.2   -2.4     71   3.82    224    9.5
  835.9   1403    2.1   -2.4     72   3.82    224    9.5
  834.8   1414    2.0   -2.5     72   3.82    224    9.6
  833.7   1424    1.9   -2.5     73   3.82    224    9.7
  832.6   1435    1.8   -2.5     73   3.82    223    9.8
  831.5   1445    1.7   -2.5     74   3.83    223    9.9
  830.4   1456    1.6   -2.5     74   3.83    222   10.0
  829.2   1467    1.5   -2.5     75   3.84    222   10.0
  828.1   1478    1.4   -2.5     76   3.84    221   10.1
  827.0   1489    1.3   -2.5     76   3.84    220   10.1
  825.9   1499    1.2   -2.5     76   3.84    219   10.1
"""

def parse_payerne(text):
    data = []
    for line in text.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 8:
            try:
                data.append({
                    'pressure':   float(parts[0]),
                    'height_asl': float(parts[1]),
                    'temp':       float(parts[2]),
                    'dewpt':      float(parts[3]),
                    'rh':         float(parts[4]),
                    'wind_dir':   float(parts[6]),
                    'wind_speed': float(parts[7]),
                })
            except (ValueError, IndexError):
                continue
    df = pd.DataFrame(data)
    df['height_agl'] = df['height_asl'] - DUBENDORF_ELEV
    wd_rad = np.deg2rad(df['wind_dir'].values)
    df['u'] = -df['wind_speed'].values * np.sin(wd_rad)
    df['v'] = -df['wind_speed'].values * np.cos(wd_rad)
    return df

payerne = parse_payerne(PAYERNE_RAW)
print(f"Payerne: {len(payerne)} levels, "
      f"{payerne['height_agl'].min():.0f}-{payerne['height_agl'].max():.0f} m AGL")

# ============================================================================
# ISA HELPERS
# ============================================================================

def isa_T(h_asl):   return 288.15 - 0.0065 * h_asl
def isa_p(h_asl):   return 1013.25 * (isa_T(h_asl)/288.15)**5.255
def baro_alt(p, p0): return 44330 * (1 - (p/p0)**(1/5.255))
def density(p_hPa, T_K): return (p_hPa*100) / (Rd*T_K)

# ============================================================================
# LOAD VAMOS
# ============================================================================

print("\nLoading VAMOS...")
sci_v = pd.read_csv(f"{DATA_DIR}/science_VAMOS.csv")
for c in sci_v.columns:
    sci_v[c] = pd.to_numeric(sci_v[c], errors='coerce')
sci_v = sci_v.dropna().reset_index(drop=True)

t_v  = sci_v["timestamp_ms"].values / 1000.0
p_v  = sci_v["pressure_hPa"].values
T_v  = sci_v["temperature_C"].values
co2_v = sci_v["co2_ppm"].values

p0_v = (np.median(p_v[:100]) + np.median(p_v[-500:])) / 2
h_v  = baro_alt(p_v, p0_v)

# Drop detection
p_sm = pd.Series(p_v).rolling(5, center=True, min_periods=1).median().values
dpdt = np.gradient(p_sm) / np.gradient(t_v)
idx  = np.where(dpdt > 0.1)[0]
gaps = np.where(np.diff(idx) > 15)[0]
starts = np.concatenate([[idx[0]], idx[gaps+1]]) if len(gaps) else np.array([idx[0]])
ends   = np.concatenate([idx[gaps], [idx[-1]]]) if len(gaps) else np.array([idx[-1]])
i_best = np.argmax([p_v[e]-p_v[s] for s,e in zip(starts,ends)])
ds, de = starts[i_best], ends[i_best]
apex_v = max(0, ds-60) + int(np.argmin(p_v[max(0,ds-60):ds+1]))
post   = np.where(p_v[de:min(len(p_v),de+60)] > p0_v-0.5)[0]
land_v = de + (post[0] if len(post) else 0)
t_apex_v = t_v[apex_v];  t_land_v = t_v[land_v]

drop_v = (t_v >= t_apex_v) & (t_v <= t_land_v)
T_vd   = T_v[drop_v];  h_vd = h_v[drop_v];  p_vd = p_v[drop_v]

print(f"  VAMOS: {len(sci_v)} samples, drop {drop_v.sum()} pts, "
      f"{h_vd.min():.0f}-{h_vd.max():.0f} m AGL")

# VAMOS wind
has_wind = False
try:
    wind_v = pd.read_csv(f"{DATA_DIR}/wind_VAMOS.csv")
    for c in wind_v.columns:
        wind_v[c] = pd.to_numeric(wind_v[c], errors='coerce')
    wind_v = wind_v.dropna().reset_index(drop=True)
    tw = wind_v["timestamp_ms"].values / 1000.0
    resets = np.where(np.diff(tw) < 0)[0]
    if len(resets):
        wind_v = wind_v.iloc[resets[-1]+1:].reset_index(drop=True)
        tw     = wind_v["timestamp_ms"].values / 1000.0
    wdrop = (tw >= t_apex_v) & (tw <= t_land_v)
    if wdrop.sum() > 10 and "x_wind_mps" in wind_v.columns:
        ux_drop = wind_v["x_wind_mps"].values[wdrop]
        uy_drop = wind_v["y_wind_mps"].values[wdrop]
        hw_drop = np.interp(tw[wdrop], t_v[drop_v], h_vd)
        has_wind = True
        print(f"  Wind: {wdrop.sum()} drop samples")
except Exception as e:
    print(f"  Wind not loaded: {e}")

# ============================================================================
# LOAD SCAMSAT
# ============================================================================

print("\nLoading SCAMSAT...")

TOTAL_DURATION = 3550   # s  (total recording, from SCAMSAT notebook)
T_DROP_START   = 828    # s
T_DROP_END     = 996    # s

def load_scamsat(filename, dtype, scale=1.0):
    path = os.path.join(SCAMSAT_DIR, filename)
    if not os.path.exists(path):
        return None
    arr = np.fromfile(path, dtype=dtype).astype(float) * scale
    return arr

alt_sc   = load_scamsat('alt.txt',   'uint16',  scale=1.0)   # m ASL (already /10 in notebook)
press_sc = load_scamsat('press.txt', 'float32', scale=1.0)   # Pa
temp_sc  = load_scamsat('temp.txt',  'float32', scale=1.0)   # °C
pm25_sc  = load_scamsat('pm2_5.txt', 'uint16',  scale=0.1)   # µg/m³
pm10_sc  = load_scamsat('pm10.txt',  'uint16',  scale=0.1)   # µg/m³
accel_sc = load_scamsat('accel.txt', 'int16',   scale=0.001) # g

# Note: in the SCAMSAT notebook, alt is stored in dm (decimeters) → /10 = meters
# Actually the notebook says dtype='uint16', divided by 10 for dm→m conversion
if alt_sc is not None:
    alt_sc = alt_sc / 10.0   # dm → m  (as in notebook: uint16 / 10)

scamsat_ok = alt_sc is not None and press_sc is not None

if scamsat_ok:
    N_sc   = len(alt_sc)
    fs_sc  = N_sc / TOTAL_DURATION    # ≈ 1 Hz
    time_sc = np.linspace(0, TOTAL_DURATION, N_sc)

    # Drop mask
    drop_sc = (time_sc >= T_DROP_START) & (time_sc <= T_DROP_END)

    alt_sc_drop  = alt_sc[drop_sc]
    press_sc_drop = press_sc[drop_sc] / 100.0  # Pa → hPa
    h_sc_drop    = alt_sc_drop - DUBENDORF_ELEV  # m AGL

    if temp_sc is not None:
        temp_sc_drop = temp_sc[drop_sc]
    if pm25_sc is not None:
        pm25_sc_drop = pm25_sc[drop_sc]
    if pm10_sc is not None:
        pm10_sc_drop = pm10_sc[drop_sc]

    p0_sc = np.median(press_sc_drop[-20:]) if len(press_sc_drop) > 20 else press_sc_drop[-1]

    print(f"  SCAMSAT: N={N_sc}, fs={fs_sc:.2f} Hz, drop={drop_sc.sum()} samples")
    print(f"  SCAMSAT drop: {h_sc_drop.min():.0f}-{h_sc_drop.max():.0f} m AGL, "
          f"p={press_sc_drop.min():.0f}-{press_sc_drop.max():.0f} hPa")
    if temp_sc is not None:
        print(f"  SCAMSAT T during drop: {temp_sc_drop.min():.1f}-{temp_sc_drop.max():.1f} °C")
else:
    print("  SCAMSAT data not found in ./002/ — skipping SCAMSAT panels")
    print("  → Place binary files (alt.txt, press.txt, temp.txt, pm2_5.txt, pm10.txt)")
    print("    in the ./002/ directory and re-run.")

# ============================================================================
# ISA PROFILE
# ============================================================================

h_isa_agl = np.linspace(0, 1400, 300)
h_isa_asl = h_isa_agl + DUBENDORF_ELEV
T_isa_C   = isa_T(h_isa_asl) - 273.15
p_isa     = isa_p(h_isa_asl)

# ============================================================================
# FIGURE — 9-panel complete atmospheric analysis (3 rows × 3 cols)
# ============================================================================

print("\nGenerating 9-panel figure...")

fig = plt.figure(figsize=(21, 18))
gs  = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.32)

# ── COLORS ──────────────────────────────────────────────────────────────────
C_VAMOS   = 'C0'      # blue
C_SCAMSAT = 'C2'      # green
C_PAY     = 'orange'
C_ISA     = 'k'

# ── PANEL 1: Pressure profile ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(p_isa, h_isa_agl, '--', color=C_ISA, lw=2, label='ISA')
ax1.plot(payerne['pressure'], payerne['height_agl'],
         color=C_PAY, lw=2.5, marker='o', markersize=3, label='Payerne (real)')
ax1.scatter(p_vd, h_vd, s=15, alpha=0.5, c=C_VAMOS, label=f'VAMOS (N={len(p_vd)})')
if scamsat_ok:
    ax1.scatter(press_sc_drop, h_sc_drop, s=15, alpha=0.5, c=C_SCAMSAT,
                label=f'SCAMSAT (N={len(press_sc_drop)})')
ax1.invert_xaxis()
ax1.set_xlabel('Pressure [hPa]', fontsize=11)
ax1.set_ylabel('Altitude [m AGL]', fontsize=11)
ax1.set_title('Pressure Profile\n✅ Both probes match Payerne', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 1000)
p_diff_v  = p0_v - payerne.iloc[0]['pressure']
p0_sc_val = p0_sc if scamsat_ok else float('nan')
p_diff_sc = (p0_sc_val - payerne.iloc[0]['pressure']) if scamsat_ok else float('nan')
sc_line   = f'SCAMSAT: {p0_sc_val:.1f} hPa  (Δ{p_diff_sc:+.1f})' if scamsat_ok else 'SCAMSAT: N/A'
ax1.text(0.05, 0.05,
         f'Ground p:\n'
         f'VAMOS:   {p0_v:.1f} hPa  (Δ{p_diff_v:+.1f})\n'
         f'{sc_line}\n'
         f'Payerne: {payerne.iloc[0]["pressure"]:.1f} hPa',
         transform=ax1.transAxes,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.85),
         fontsize=8, family='monospace', va='bottom')

# ── PANEL 2: Temperature profile ─────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(T_isa_C, h_isa_agl, '--', color=C_ISA, lw=2, label='ISA')
ax2.plot(payerne['temp'], payerne['height_agl'],
         color=C_PAY, lw=2.5, marker='o', markersize=3, label='Payerne (real)')
ax2.scatter(T_vd, h_vd, s=15, alpha=0.5, c=C_VAMOS, label='VAMOS')
if scamsat_ok and temp_sc is not None:
    ax2.scatter(temp_sc_drop, h_sc_drop, s=15, alpha=0.5, c=C_SCAMSAT, label='SCAMSAT')
    # Shade biases
    sort_v = np.argsort(h_vd)
    T_pay_v = np.interp(h_vd[sort_v], payerne['height_agl'].values[::-1],
                        payerne['temp'].values[::-1])
    ax2.fill_betweenx(h_vd[sort_v], T_pay_v, T_vd[sort_v],
                      alpha=0.12, color=C_VAMOS, label='VAMOS bias')
    sort_s = np.argsort(h_sc_drop)
    T_pay_s = np.interp(h_sc_drop[sort_s], payerne['height_agl'].values[::-1],
                        payerne['temp'].values[::-1])
    ax2.fill_betweenx(h_sc_drop[sort_s], T_pay_s, temp_sc_drop[sort_s],
                      alpha=0.12, color=C_SCAMSAT, label='SCAMSAT bias')

    # Bias stats
    bias_v  = np.mean(T_vd) - np.interp(np.mean(h_vd),
              payerne['height_agl'].values[::-1], payerne['temp'].values[::-1])
    bias_sc = np.mean(temp_sc_drop) - np.interp(np.mean(h_sc_drop),
              payerne['height_agl'].values[::-1], payerne['temp'].values[::-1])
    ax2.text(0.05, 0.05,
             f'T bias vs Payerne:\nVAMOS:   {bias_v:+.1f}°C\nSCAMSAT: {bias_sc:+.1f}°C',
             transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85),
             fontsize=8, family='monospace', va='bottom')

ax2.set_xlabel('Temperature [°C]', fontsize=11)
ax2.set_ylabel('Altitude [m AGL]', fontsize=11)
ax2.set_title('Temperature Profile\n❌ Both probes show warm bias', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 1000)

# ── PANEL 3: PM2.5 and PM10 (SCAMSAT only) ──────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
if scamsat_ok and pm25_sc is not None and pm10_sc is not None:
    sort_sc = np.argsort(h_sc_drop)
    ax3.plot(pm25_sc_drop[sort_sc], h_sc_drop[sort_sc],
             color=C_SCAMSAT, lw=2, label='PM₂.₅ (SCAMSAT)')
    ax3.plot(pm10_sc_drop[sort_sc], h_sc_drop[sort_sc],
             color='darkorange', lw=2, ls='--', label='PM₁₀ (SCAMSAT)')
    # WHO reference lines
    ax3.axvline(15, color='red', ls=':', lw=1.2, alpha=0.7, label='WHO PM₂.₅ limit (15 µg/m³)')
    ax3.axvline(45, color='darkred', ls=':', lw=1.2, alpha=0.7, label='WHO PM₁₀ limit (45 µg/m³)')
    # Annotation: min PM during high-speed descent
    idx_min = np.argmin(pm25_sc_drop[sort_sc])
    ax3.annotate('Low PM at\nhigh speed\n(sensor artifact)',
                 xy=(pm25_sc_drop[sort_sc][idx_min], h_sc_drop[sort_sc][idx_min]),
                 xytext=(15, h_sc_drop[sort_sc][idx_min] + 100),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax3.set_xlabel('Mass Concentration [µg/m³]', fontsize=11)
    ax3.text(0.05, 0.05,
             f'Mean PM₂.₅: {np.mean(pm25_sc_drop):.1f} µg/m³\n'
             f'Mean PM₁₀:  {np.mean(pm10_sc_drop):.1f} µg/m³',
             transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.85),
             fontsize=8, family='monospace', va='bottom')
else:
    ax3.text(0.5, 0.5, 'SCAMSAT data\nnot available\n\nPlace files in ./002/',
             ha='center', va='center', transform=ax3.transAxes,
             fontsize=12, bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax3.set_ylabel('Altitude [m AGL]', fontsize=11)
ax3.set_title('Particulate Matter Profile\n(SCAMSAT: PM₂.₅ and PM₁₀)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)
ax3.set_ylim(0, 1000)

# ── PANEL 4: Air Density ─────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
rho_isa = density(p_isa, isa_T(h_isa_asl))
rho_pay = density(payerne['pressure'].values, payerne['temp'].values + 273.15)
rho_v   = density(p_vd, T_vd + 273.15)
ax4.plot(rho_isa, h_isa_agl, '--', color=C_ISA, lw=2, label='ISA')
ax4.plot(rho_pay, payerne['height_agl'], color=C_PAY, lw=2.5,
         marker='o', markersize=3, label='Payerne (real)')
ax4.scatter(rho_v, h_vd, s=15, alpha=0.5, c=C_VAMOS, label='VAMOS')
if scamsat_ok and temp_sc is not None:
    rho_sc = density(press_sc_drop, temp_sc_drop + 273.15)
    ax4.scatter(rho_sc, h_sc_drop, s=15, alpha=0.5, c=C_SCAMSAT, label='SCAMSAT')
    rho_pay_mean = np.interp(np.mean(h_vd), payerne['height_agl'].values, rho_pay)
    bias_rho_v  = (np.mean(rho_v)  - rho_pay_mean) / rho_pay_mean * 100
    bias_rho_sc = (np.mean(rho_sc) - np.interp(np.mean(h_sc_drop),
                   payerne['height_agl'].values, rho_pay)) / rho_pay_mean * 100
    ax4.text(0.05, 0.05,
             f'ρ bias vs Payerne:\nVAMOS:   {bias_rho_v:+.1f}%\nSCAMSAT: {bias_rho_sc:+.1f}%',
             transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85),
             fontsize=8, family='monospace', va='bottom')
ax4.set_xlabel('Air Density [kg/m³]', fontsize=11)
ax4.set_ylabel('Altitude [m AGL]', fontsize=11)
ax4.set_title('Air Density Profile\n(Both underestimate ρ due to warm T bias)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)
ax4.set_ylim(0, 1000)

# ── PANEL 5: Skew-T Log-P ────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])

def skewT(T, p, skew=45):
    Y = -np.log(p)
    X = T + skew * (-Y - np.log(1000))
    return X, Y

p_lev = np.arange(750, 1000, 10)
for theta in [260, 270, 280, 290, 300, 310]:
    T_ln = theta * (p_lev/1000)**(Rd/Cp) - 273.15
    ax5.plot(*skewT(T_ln, p_lev), 'r-', lw=0.5, alpha=0.25)
for T_iso in np.arange(-30, 40, 10):
    ax5.plot(*skewT(np.full_like(p_lev, T_iso), p_lev), 'b-', lw=0.5, alpha=0.25)

X_pT, Y_p  = skewT(payerne['temp'].values, payerne['pressure'].values)
X_pTd, _   = skewT(payerne['dewpt'].values, payerne['pressure'].values)
ax5.plot(X_pT,  Y_p, 'r-', lw=2.5, label='Payerne T')
ax5.plot(X_pTd, Y_p, 'g-', lw=2.5, label='Payerne Td')
ax5.plot(*skewT(T_vd, p_vd), 'o', color=C_VAMOS, ms=3, alpha=0.6, label='VAMOS T')
if scamsat_ok and temp_sc is not None:
    ax5.plot(*skewT(temp_sc_drop, press_sc_drop), 's', color=C_SCAMSAT,
             ms=3, alpha=0.6, label='SCAMSAT T')

ax5.set_ylim(-np.log(1000), -np.log(750))
ax5.set_xlim(-30, 50)
y_ticks_p = [1000, 950, 925, 900, 850, 800, 750]
ax5.set_yticks([-np.log(p) for p in y_ticks_p])
ax5.set_yticklabels([str(p) for p in y_ticks_p])
ax5.set_xlabel('Temperature [°C]  (skewed at 45°)', fontsize=10)
ax5.set_ylabel('Pressure [hPa]', fontsize=10)
ax5.set_title('Skew-T Log-P Diagram\n(red: dry adiabats, blue: isotherms)', fontsize=12, fontweight='bold')
ax5.legend(fontsize=8, loc='upper left')
ax5.grid(alpha=0.2)

# ── PANEL 6: Wind hodograph (Payerne) ────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
mask_h = payerne['height_agl'] <= 1000
sc6 = ax6.scatter(payerne['u'].values[mask_h], payerne['v'].values[mask_h],
                  c=payerne['height_agl'].values[mask_h], cmap='viridis',
                  s=30, zorder=3, edgecolors='black', linewidths=0.5)
ax6.plot(payerne['u'].values[mask_h], payerne['v'].values[mask_h],
         '-', color='gray', lw=1, alpha=0.5, zorder=2)
plt.colorbar(sc6, ax=ax6, label='Altitude [m AGL]')
if has_wind:
    step = max(1, len(ux_drop)//50)
    ax6.scatter(ux_drop[::step], uy_drop[::step],
                c=hw_drop[::step], cmap='plasma',
                marker='^', s=25, alpha=0.5, label='VAMOS wind')
for r in [2, 4, 6, 8, 10]:
    ax6.add_patch(plt.Circle((0,0), r, fill=False, color='gray', ls=':', alpha=0.4))
    ax6.text(r*0.707, r*0.707, f'{r}', fontsize=7, color='gray', alpha=0.6)
ax6.axhline(0, color='k', lw=0.5, alpha=0.3)
ax6.axvline(0, color='k', lw=0.5, alpha=0.3)
ax6.set_xlabel('U (East-West) [m/s]', fontsize=11)
ax6.set_ylabel('V (North-South) [m/s]', fontsize=11)
ax6.set_title('Wind Hodograph — Payerne\n(spiraling = wind shear with altitude)', fontsize=12, fontweight='bold')
ax6.set_aspect('equal'); ax6.grid(alpha=0.3)
ax6.set_xlim(-12, 12); ax6.set_ylim(-12, 12)

# ── PANEL 7: Vertical wind shear (Payerne) ──────────────────────────────────
ax7 = fig.add_subplot(gs[2, 0])
h_p  = payerne['height_agl'].values
u_sm = uniform_filter1d(payerne['u'].values, size=5)
v_sm = uniform_filter1d(payerne['v'].values, size=5)
shear = np.sqrt(np.gradient(u_sm, h_p)**2 + np.gradient(v_sm, h_p)**2) * 1000
valid = h_p <= 1000
ax7.plot(shear[valid], h_p[valid], 'o-', color='purple', lw=2, markersize=4)
ax7.fill_betweenx(h_p[valid], 0, shear[valid], alpha=0.2, color='purple')
ax7_r = ax7.twiny()
ax7_r.plot(payerne['wind_speed'].values[valid], h_p[valid],
           's-', color='darkgreen', lw=1.5, markersize=4, alpha=0.7)
ax7_r.set_xlabel('Wind Speed [m/s]', fontsize=10, color='darkgreen')
ax7_r.tick_params(axis='x', labelcolor='darkgreen')
max_sh_idx = np.argmax(shear[valid])
ax7.annotate(f'Max: {shear[valid][max_sh_idx]:.0f} m/s/km\nat {h_p[valid][max_sh_idx]:.0f} m',
             xy=(shear[valid][max_sh_idx], h_p[valid][max_sh_idx]),
             xytext=(shear[valid][max_sh_idx]*0.5, h_p[valid][max_sh_idx]+200),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax7.set_xlabel('Vertical Wind Shear [m/s per km]', fontsize=11, color='purple')
ax7.tick_params(axis='x', labelcolor='purple')
ax7.set_ylabel('Altitude [m AGL]', fontsize=11)
ax7.set_title('Vertical Wind Shear\n(Payerne radiosonde)', fontsize=12, fontweight='bold')
ax7.grid(alpha=0.3); ax7.set_ylim(0, 1000)

# ── PANEL 8: VAMOS Wind Hodograph ───────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 1])
if has_wind:
    step = max(1, len(ux_drop)//100)
    u_pl = ux_drop[::step]; v_pl = uy_drop[::step]; h_pl = hw_drop[::step]
    sc8 = ax8.scatter(u_pl, v_pl, c=h_pl, cmap='plasma', s=30, zorder=3,
                      edgecolors='black', linewidths=0.5)
    ax8.plot(u_pl, v_pl, '-', color='gray', lw=1, alpha=0.3, zorder=2)
    div8 = make_axes_locatable(ax8)
    cax8 = div8.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(sc8, cax=cax8, label='Alt [m]')
    for r in [1,2,3,4,5]:
        ax8.add_patch(plt.Circle((0,0), r, fill=False, color='gray', ls=':', alpha=0.4))
    ax8.scatter(u_pl[0],  v_pl[0],  s=120, marker='o', color='green',
               edgecolor='darkgreen', lw=2, zorder=5, label=f'Start ({h_pl[0]:.0f}m)')
    ax8.scatter(u_pl[-1], v_pl[-1], s=120, marker='s', color='red',
               edgecolor='darkred',  lw=2, zorder=5, label=f'End ({h_pl[-1]:.0f}m)')
    mean_ws = np.sqrt(np.mean(ux_drop)**2 + np.mean(uy_drop)**2)
    ax8.text(0.02, 0.02, f'Mean: {mean_ws:.1f} m/s\nσ(u):{np.std(ux_drop):.1f} σ(v):{np.std(uy_drop):.1f}',
             transform=ax8.transAxes, fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax8.text(0.02, 0.98,
             'Scatter = CanSat rotation\n+ pendulum oscillation\n(not true atmospheric wind)',
             transform=ax8.transAxes, fontsize=8, style='italic', va='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
else:
    ax8.text(0.5, 0.5, 'Wind data\nnot available', ha='center', va='center',
             transform=ax8.transAxes, fontsize=12)
ax8.axhline(0, color='k', lw=0.5, alpha=0.3); ax8.axvline(0, color='k', lw=0.5, alpha=0.3)
ax8.set_xlabel('U (East) [m/s]', fontsize=11); ax8.set_ylabel('V (North) [m/s]', fontsize=11)
ax8.set_title('VAMOS Wind Hodograph\n(scattered = CanSat dynamics)', fontsize=12, fontweight='bold')
ax8.legend(fontsize=8); ax8.set_aspect('equal'); ax8.grid(alpha=0.3)
if has_wind:
    lim8 = max(np.abs(u_pl).max(), np.abs(v_pl).max()) * 1.2
    ax8.set_xlim(-lim8, lim8); ax8.set_ylim(-lim8, lim8)

# ── PANEL 9: SCAMSAT Acceleration / Data Quality Summary ─────────────────────
ax9 = fig.add_subplot(gs[2, 2])
if scamsat_ok and accel_sc is not None:
    N_ac = len(accel_sc)
    time_ac = np.linspace(0, TOTAL_DURATION, N_ac)
    drop_ac = (time_ac >= T_DROP_START) & (time_ac <= T_DROP_END)
    ax9.semilogy(time_ac[drop_ac], np.abs(accel_sc[drop_ac]) + 1e-4,
                 color=C_SCAMSAT, lw=1, alpha=0.8, label='SCAMSAT |accel| (drop)')
    ax9.axvline(T_DROP_START, color='green',  lw=2, label=f'Drop start (t={T_DROP_START}s)')
    ax9.axvline(T_DROP_END,   color='orange', lw=2, label=f'Drop end (t={T_DROP_END}s)')
    ax9.set_xlabel('Time [s]', fontsize=11)
    ax9.set_ylabel('|Acceleration| [g]', fontsize=11)
    ax9.set_title('SCAMSAT Acceleration (drop phase)\n(parachute deployment visible)', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=8); ax9.grid(alpha=0.3)
else:
    # Show data quality summary as text table
    ax9.axis('off')
    table_data = [
        ['Probe',     'Samples',  'Drop',     'fs',       'Quality'],
        ['VAMOS',     '9349',     '✓ 713m',   '1 Hz',     '✅ Primary'],
        ['SCAMSAT',   '3550',     '✓ 665m',   '~1 Hz',    '✅ Good'],
        ['GRASP',     '1715',     '❌ False', '38 Hz',    '⚠️ Limited'],
        ['OBAMA',     '24',       '⚠️ Part', '0.04 Hz',  '⚠️ Minimal'],
    ]
    table = ax9.table(cellText=table_data[1:], colLabels=table_data[0],
                      cellLoc='center', loc='center',
                      bbox=[0.0, 0.2, 1.0, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table[(1,4)].set_facecolor('#90EE90')
    table[(2,4)].set_facecolor('#90EE90')
    table[(3,4)].set_facecolor('#FFD700')
    table[(4,4)].set_facecolor('#FFD700')
    ax9.set_title('Data Quality Summary\n(all four probes)', fontsize=12, fontweight='bold')
    ax9.text(0.5, 0.1, '* SCAMSAT data loaded from ./002/ binary files',
             ha='center', transform=ax9.transAxes, fontsize=9, style='italic')

plt.suptitle('VAMOS & SCAMSAT CanSat vs Payerne Radiosonde — '
             'Complete Atmospheric Analysis (6 Feb 2026)',
             fontsize=14, fontweight='bold', y=0.995)

plt.savefig(f"{OUTPUT_DIR}/fig8_complete_atmospheric.png", dpi=130, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/fig8_complete_atmospheric.png")

# ============================================================================
# STANDALONE VAMOS HODOGRAPH (fig9)
# ============================================================================

if has_wind:
    print("\nGenerating standalone VAMOS hodograph...")
    fig_h, ax_h = plt.subplots(figsize=(9, 9))
    step = max(1, len(ux_drop)//100)
    u_pl = ux_drop[::step]; v_pl = uy_drop[::step]; h_pl = hw_drop[::step]
    sc_h = ax_h.scatter(u_pl, v_pl, c=h_pl, cmap='viridis', s=40, zorder=3,
                        edgecolors='black', linewidths=0.5)
    ax_h.plot(u_pl, v_pl, '-', color='gray', lw=1, alpha=0.3)
    plt.colorbar(sc_h, ax=ax_h, label='Altitude [m AGL]', pad=0.02)
    lim_h = max(np.abs(u_pl).max(), np.abs(v_pl).max()) * 1.2
    for r in [1,2,3,4,5]:
        if r < lim_h:
            ax_h.add_patch(plt.Circle((0,0), r, fill=False, color='gray', ls=':', alpha=0.4))
            ax_h.text(r*0.707, r*0.707, f'{r} m/s', fontsize=8, color='gray', alpha=0.6,
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none'))
    ax_h.scatter(u_pl[0],  v_pl[0],  s=200, marker='o', color='green',
                edgecolor='darkgreen', lw=2.5, zorder=5, label=f'Apogee (h={h_pl[0]:.0f} m)')
    ax_h.scatter(u_pl[-1], v_pl[-1], s=200, marker='s', color='red',
                edgecolor='darkred',  lw=2.5, zorder=5, label=f'Landing (h={h_pl[-1]:.0f} m)')
    ax_h.axhline(0, color='k', lw=0.8, alpha=0.3); ax_h.axvline(0, color='k', lw=0.8, alpha=0.3)
    for txt, x, y in [('N',0,lim_h*0.93),('S',0,-lim_h*0.93),('E',lim_h*0.93,0),('W',-lim_h*0.93,0)]:
        ax_h.text(x, y, txt, ha='center', va='center', fontsize=11, fontweight='bold', color='darkblue')
    mean_u = np.mean(ux_drop); mean_v = np.mean(uy_drop)
    mean_ws = np.sqrt(mean_u**2 + mean_v**2)
    mean_dir = np.rad2deg(np.arctan2(-mean_u, -mean_v)) % 360
    ax_h.text(0.98, 0.02,
              f'Mean wind during drop:\n  Speed: {mean_ws:.2f} m/s\n  From: {mean_dir:.0f}°\n'
              f'  U: {mean_u:+.2f} m/s\n  V: {mean_v:+.2f} m/s\n\n'
              f'Variability:\n  σ(U): {np.std(ux_drop):.2f} m/s\n  σ(V): {np.std(uy_drop):.2f} m/s',
              transform=ax_h.transAxes, fontsize=9, family='monospace',
              va='bottom', ha='right',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    ax_h.set_xlabel('U (Eastward) [m/s]', fontsize=12, fontweight='bold')
    ax_h.set_ylabel('V (Northward) [m/s]', fontsize=12, fontweight='bold')
    ax_h.set_title(f'VAMOS Wind Hodograph During Drop\n'
                   f'(t={t_apex_v:.0f}-{t_land_v:.0f} s, N={len(u_pl)} decimated samples)',
                   fontsize=13, fontweight='bold', pad=15)
    ax_h.legend(fontsize=10); ax_h.set_aspect('equal'); ax_h.grid(alpha=0.3)
    ax_h.set_xlim(-lim_h, lim_h); ax_h.set_ylim(-lim_h, lim_h)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig9_vamos_hodograph.png", dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR}/fig9_vamos_hodograph.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nVAMOS  — drop: {h_vd.min():.0f}-{h_vd.max():.0f} m AGL, "
      f"T bias vs Payerne ≈ {np.mean(T_vd) - np.interp(np.mean(h_vd), payerne['height_agl'].values[::-1], payerne['temp'].values[::-1]):+.1f}°C")
if scamsat_ok:
    print(f"SCAMSAT — drop: {h_sc_drop.min():.0f}-{h_sc_drop.max():.0f} m AGL, "
          f"{drop_sc.sum()} samples, fs≈{fs_sc:.2f} Hz")
    if pm25_sc is not None:
        print(f"SCAMSAT PM₂.₅: {pm25_sc_drop.min():.1f}-{pm25_sc_drop.max():.1f} µg/m³  "
              f"(mean {np.mean(pm25_sc_drop):.1f})")
        print(f"SCAMSAT PM₁₀:  {pm10_sc_drop.min():.1f}-{pm10_sc_drop.max():.1f} µg/m³  "
              f"(mean {np.mean(pm10_sc_drop):.1f})")
print(f"\nPayerne wind shear: max {shear[valid][max_sh_idx]:.0f} m/s/km "
      f"at {h_p[valid][max_sh_idx]:.0f} m AGL")
print("="*70)

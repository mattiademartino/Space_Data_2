"""
CanSat Complete Atmospheric Analysis
=====================================

Produces 4 professional atmospheric plots:
1. Pressure profile (VAMOS vs Payerne vs ISA)
2. Air density profile (comparative)
3. Skew-T diagram (professional radiosonde style)
4. Wind shear profile (hodograph + vertical shear)

Uses REAL Payerne radiosonde data from 6 Feb 2026.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "./figures"
DATA_DIR = "./Cansat data utili"
DUBENDORF_ELEV = 448   # m ASL
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Physical constants
Rd = 287.05           # J/(kg·K) specific gas constant for dry air
Rv = 461.5            # J/(kg·K) water vapor
g = 9.80665           # m/s²
Cp = 1004.0           # J/(kg·K) specific heat at constant pressure
Lv = 2.5e6            # J/kg latent heat of vaporization

# ============================================================================
# PAYERNE RADIOSONDE DATA (6 FEB 2026)
# ============================================================================

PAYERNE_DATA_TEXT = """
  935.9    491    1.9    0.9     93   4.36      4    0.3  280.3  292.5  281.0
  933.8    509    1.1    1.0     99   4.39     45    0.9  279.7  292.0  280.4
  932.7    519    0.9    0.9    100   4.37     48    1.2  279.6  291.8  280.3
  931.6    528    0.8    0.7     99   4.33     45    1.2  279.6  291.7  280.3
  930.4    538    0.8    0.7     99   4.31     40    1.1  279.6  291.7  280.3
  929.1    550    0.7    0.6     99   4.29     35    1.1  279.6  291.6  280.3
  927.8    561    0.6    0.5    100   4.28     29    1.2  279.6  291.6  280.4
  926.5    572    0.5    0.5    100   4.27     24    1.2  279.7  291.6  280.4
  925.2    584    0.4    0.4    100   4.26     19    1.2  279.7  291.6  280.5
  923.9    595    0.4    0.4    100   4.25     15    1.3  279.8  291.7  280.5
  922.6    606    0.3    0.3    100   4.24     12    1.3  279.9  291.7  280.6
  921.4    617    0.3    0.3    100   4.24     10    1.3  279.9  291.8  280.6
  920.4    626    0.2    0.2    100   4.20      8    1.3  279.9  291.7  280.6
  919.1    637    0.1    0.1    100   4.18      7    1.3  279.9  291.6  280.6
  917.9    647    0.1    0.1    100   4.19      8    1.3  280.0  291.7  280.7
  916.5    660    0.0    0.0    100   4.18      9    1.2  280.1  291.8  280.8
  914.9    674   -0.0   -0.0    100   4.17     12    1.1  280.2  291.9  280.9
  913.5    686   -0.1   -0.1    100   4.14     16    0.9  280.2  291.8  280.9
  912.3    696   -0.2   -0.2    100   4.14     23    0.8  280.2  291.9  280.9
  910.9    709   -0.2   -0.2    100   4.13     33    0.6  280.3  291.9  281.0
  909.4    722   -0.3   -0.3    100   4.11     50    0.5  280.3  291.9  281.0
  907.4    740   -0.4   -0.4    100   4.09     90    0.4  280.4  291.9  281.1
  905.6    755   -0.5   -0.5    100   4.07    123    0.5  280.5  291.9  281.2
  904.0    770   -0.6   -0.6    100   4.05    140    0.7  280.5  291.9  281.2
  902.3    784   -0.7   -0.8    100   4.01    151    0.9  280.6  291.9  281.2
  900.5    801   -0.2   -2.6     84   3.52    159    1.1  281.3  291.2  281.8
  899.3    811    0.2   -3.0     79   3.39    163    1.2  281.7  291.4  282.3
  898.3    821    1.0   -2.8     76   3.47    168    1.3  282.7  292.7  283.3
  897.2    830    1.3   -3.7     70   3.24    172    1.5  283.0  292.4  283.6
  896.1    840    1.8   -3.4     68   3.32    176    1.6  283.7  293.3  284.3
  894.9    850    2.1   -3.1     68   3.38    180    1.8  284.1  293.9  284.7
  893.3    865    2.5   -2.8     68   3.47    185    2.0  284.7  294.8  285.3
  892.0    877    2.8   -2.5     68   3.57    188    2.2  285.1  295.5  285.8
  890.4    892    3.0   -2.5     67   3.57    192    2.6  285.5  295.8  286.1
  889.3    901    3.1   -2.5     67   3.58    194    2.8  285.7  296.1  286.3
  888.1    913    3.1   -2.5     67   3.59    195    3.0  285.8  296.2  286.4
  887.0    923    3.1   -2.5     67   3.59    197    3.2  285.9  296.3  286.5
  886.0    932    3.0   -2.5     67   3.58    198    3.5  285.9  296.3  286.5
  885.0    941    3.0   -2.7     66   3.54    200    3.7  285.9  296.2  286.6
  884.1    949    2.9   -2.8     66   3.50    201    3.9  286.0  296.2  286.6
  883.1    958    2.8   -2.9     66   3.51    202    4.0  286.0  296.2  286.6
  882.1    968    2.8   -3.0     65   3.47    203    4.2  286.0  296.1  286.6
  881.0    978    2.8   -3.2     65   3.44    204    4.4  286.1  296.1  286.7
  879.9    988    2.7   -3.3     65   3.40    205    4.6  286.1  296.0  286.7
  878.7    998    2.6   -3.8     62   3.28    206    4.7  286.2  295.7  286.7
  877.5   1010    2.7   -4.1     61   3.22    207    4.9  286.3  295.7  286.9
  876.3   1021    2.6   -4.1     62   3.23    208    5.0  286.3  295.8  286.9
  875.0   1032    2.5   -4.1     62   3.23    209    5.2  286.3  295.8  286.9
  873.8   1044    2.4   -4.3     61   3.18    210    5.3  286.4  295.7  286.9
  872.5   1056    2.5   -5.2     57   2.98    210    5.4  286.6  295.4  287.1
  871.4   1066    2.7   -5.6     54   2.88    210    5.5  286.9  295.4  287.4
  870.2   1077    2.7   -5.5     55   2.90    210    5.6  287.0  295.6  287.5
  869.2   1087    2.8   -5.6     54   2.89    211    5.7  287.2  295.7  287.7
  868.1   1096    2.8   -5.6     54   2.89    210    5.7  287.3  295.8  287.8
  866.9   1108    3.0   -5.1     55   3.03    210    5.8  287.7  296.6  288.2
  865.6   1120    3.5   -4.5     56   3.15    210    5.8  288.3  297.6  288.8
  864.5   1130    3.5   -4.0     58   3.29    210    5.8  288.4  298.1  289.0
  863.4   1140    3.5   -3.3     61   3.48    209    5.8  288.5  298.7  289.1
  862.3   1151    3.4   -3.3     62   3.48    209    5.9  288.5  298.7  289.1
  861.4   1160    3.3   -3.2     62   3.49    209    5.9  288.5  298.8  289.1
  860.6   1167    3.2   -3.2     63   3.51    209    5.9  288.5  298.8  289.1
  859.8   1175    3.2   -3.2     63   3.52    209    6.0  288.5  298.8  289.1
  858.7   1185    3.1   -3.2     63   3.52    210    6.1  288.5  298.8  289.1
  857.6   1196    3.0   -3.1     64   3.55    211    6.2  288.5  298.9  289.1
  856.5   1206    2.9   -2.8     66   3.64    211    6.4  288.6  299.2  289.2
  855.5   1215    2.8   -2.6     67   3.68    212    6.6  288.6  299.4  289.2
  854.4   1226    2.8   -2.6     68   3.70    214    6.8  288.6  299.5  289.3
  853.2   1237    2.7   -2.5     69   3.73    215    7.0  288.6  299.6  289.3
  852.1   1248    2.6   -2.5     69   3.73    216    7.2  288.7  299.6  289.3
  851.0   1258    2.6   -2.5     69   3.72    217    7.5  288.7  299.6  289.4
  850.0   1268    2.5   -2.5     70   3.75    218    7.6  288.7  299.7  289.4
  849.3   1274    2.4   -2.4     70   3.77    219    7.8  288.7  299.8  289.4
  848.2   1285    2.4   -2.2     72   3.83    219    8.0  288.8  300.0  289.5
  847.1   1295    2.3   -2.1     72   3.85    220    8.3  288.9  300.1  289.5
  846.0   1306    2.3   -2.1     73   3.88    221    8.5  288.9  300.3  289.6
  844.8   1317    2.2   -1.9     74   3.93    221    8.6  289.0  300.5  289.7
  843.7   1328    2.2   -1.9     74   3.94    222    8.8  289.1  300.6  289.8
  842.6   1338    2.2   -2.0     74   3.92    222    8.9  289.2  300.7  289.9
  841.5   1348    2.3   -2.2     72   3.87    223    9.1  289.3  300.7  290.0
  840.4   1359    2.2   -2.3     72   3.84    223    9.2  289.4  300.7  290.1
  839.2   1371    2.2   -2.3     72   3.84    223    9.3  289.5  300.7  290.1
  838.1   1382    2.2   -2.4     72   3.83    223    9.4  289.6  300.8  290.3
  837.0   1392    2.2   -2.4     71   3.82    224    9.5  289.7  300.9  290.4
  835.9   1403    2.1   -2.4     72   3.82    224    9.5  289.7  300.9  290.4
  834.8   1414    2.0   -2.5     72   3.82    224    9.6  289.7  301.0  290.4
  833.7   1424    1.9   -2.5     73   3.82    224    9.7  289.7  301.0  290.4
  832.6   1435    1.8   -2.5     73   3.82    223    9.8  289.7  301.0  290.4
  831.5   1445    1.7   -2.5     74   3.83    223    9.9  289.7  301.0  290.4
  830.4   1456    1.6   -2.5     74   3.83    222   10.0  289.7  301.0  290.4
  829.2   1467    1.5   -2.5     75   3.84    222   10.0  289.7  301.0  290.4
  828.1   1478    1.4   -2.5     76   3.84    221   10.1  289.7  301.0  290.4
  827.0   1489    1.3   -2.5     76   3.84    220   10.1  289.7  301.0  290.4
  825.9   1499    1.2   -2.5     76   3.84    219   10.1  289.7  301.0  290.4
"""

def parse_payerne(text):
    """Parse Wyoming radiosonde text format."""
    data = []
    for line in text.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 8:
            try:
                data.append({
                    'pressure': float(parts[0]),
                    'height_asl': float(parts[1]),
                    'temp': float(parts[2]),
                    'dewpt': float(parts[3]),
                    'rh': float(parts[4]),
                    'mixr': float(parts[5]),
                    'wind_dir': float(parts[6]),
                    'wind_speed': float(parts[7]),
                })
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(data)

payerne = parse_payerne(PAYERNE_DATA_TEXT)
payerne['height_agl'] = payerne['height_asl'] - DUBENDORF_ELEV

# Compute u, v components from wind dir/speed
# Meteorological convention: dir is FROM (so we need to reverse)
wind_dir_rad = np.deg2rad(payerne['wind_dir'].values)
payerne['u'] = -payerne['wind_speed'].values * np.sin(wind_dir_rad)  # eastward
payerne['v'] = -payerne['wind_speed'].values * np.cos(wind_dir_rad)  # northward

print("="*70)
print(f"Payerne radiosonde loaded: {len(payerne)} levels")
print("="*70)

# ============================================================================
# ISA REFERENCE
# ============================================================================

def isa_temperature(h_asl):
    return 288.15 - 0.0065 * h_asl

def isa_pressure(h_asl):
    T_h = isa_temperature(h_asl)
    return 1013.25 * (T_h / 288.15) ** 5.255

def barometric_altitude(p, p0):
    return 44330 * (1 - (p / p0) ** (1/5.255))

def air_density(p_hpa, T_K):
    """Air density from ideal gas law [kg/m³]."""
    return (p_hpa * 100) / (Rd * T_K)

def potential_temperature(T_K, p_hpa, p0=1000):
    """Potential temperature θ = T × (p0/p)^(R/Cp)."""
    return T_K * (p0 / p_hpa) ** (Rd / Cp)

# ============================================================================
# LOAD VAMOS DATA
# ============================================================================

print("\nLoading VAMOS data...")
sci_v = pd.read_csv(f"{DATA_DIR}/science_VAMOS.csv")
for c in sci_v.columns:
    sci_v[c] = pd.to_numeric(sci_v[c], errors="coerce")
sci_v = sci_v.dropna().reset_index(drop=True)

t_v = sci_v["timestamp_ms"].values / 1000.0
p_v = sci_v["pressure_hPa"].values
T_v = sci_v["temperature_C"].values

# Try to load wind data for VAMOS
try:
    wind_v = pd.read_csv(f"{DATA_DIR}/wind_VAMOS.csv")
    for c in wind_v.columns:
        wind_v[c] = pd.to_numeric(wind_v[c], errors="coerce")
    wind_v = wind_v.dropna().reset_index(drop=True)
    
    # Handle timestamp resets
    tw_v = wind_v["timestamp_ms"].values / 1000.0
    dt_w = np.diff(tw_v)
    reset_idx = np.where(dt_w < 0)[0]
    if len(reset_idx):
        last_reset = reset_idx[-1] + 1
        wind_v = wind_v.iloc[last_reset:].reset_index(drop=True)
    
    has_wind = True
    print(f"  Wind data: {len(wind_v)} samples")
except Exception as e:
    has_wind = False
    print(f"  No wind data: {e}")

# Ground reference
p0_vamos = (np.median(p_v[:100]) + np.median(p_v[-500:])) / 2
h_vamos_agl = barometric_altitude(p_v, p0_vamos)

# Identify drop phase
p_smooth = pd.Series(p_v).rolling(5, center=True, min_periods=1).median().values
dpdt = np.gradient(p_smooth) / np.gradient(t_v)
above = dpdt > 0.1
idx = np.where(above)[0]
gaps = np.where(np.diff(idx) > 15)[0]
if len(gaps):
    starts = np.concatenate([[idx[0]], idx[gaps+1]])
    ends = np.concatenate([idx[gaps], [idx[-1]]])
else:
    starts = np.array([idx[0]])
    ends = np.array([idx[-1]])
dps = np.array([p_v[e] - p_v[s] for s, e in zip(starts, ends)])
i_best = np.argmax(dps)
ds, de = starts[i_best], ends[i_best]
look_back = max(0, ds - 60)
apogee_idx = look_back + int(np.argmin(p_v[look_back:ds+1]))
look_fwd = min(len(p_v), de + 60)
post_ground = np.where(p_v[de:look_fwd] > p0_vamos - 0.5)[0]
landing_idx = de + (post_ground[0] if len(post_ground) else 0)
t_apogee = t_v[apogee_idx]
t_landing = t_v[landing_idx]

drop_mask = (t_v >= t_apogee) & (t_v <= t_landing)
T_vamos_drop = T_v[drop_mask]
h_vamos_drop = h_vamos_agl[drop_mask]
p_vamos_drop = p_v[drop_mask]
t_vamos_drop = t_v[drop_mask]

print(f"  Drop phase: {drop_mask.sum()} samples, {h_vamos_drop.min():.0f}-{h_vamos_drop.max():.0f} m AGL")

# Extract VAMOS wind during drop (if available)
if has_wind:
    tw = wind_v["timestamp_ms"].values / 1000.0
    wind_drop_mask = (tw >= t_apogee) & (tw <= t_landing)
    if wind_drop_mask.sum() > 10:
        ws_drop = wind_v["wind_speed"].values[wind_drop_mask]
        if "x_wind_mps" in wind_v.columns and "y_wind_mps" in wind_v.columns:
            ux_drop = wind_v["x_wind_mps"].values[wind_drop_mask]
            uy_drop = wind_v["y_wind_mps"].values[wind_drop_mask]
        else:
            ux_drop = wind_v["x_wind_acc"].values[wind_drop_mask]
            uy_drop = wind_v["y_wind_acc"].values[wind_drop_mask]
        tw_drop = tw[wind_drop_mask]
        
        # Interpolate altitude to wind timestamps
        h_wind_drop = np.interp(tw_drop, t_vamos_drop, h_vamos_drop)

# ============================================================================
# FIGURE 1: Comprehensive 6-panel atmospheric analysis
# ============================================================================

print("\nGenerating comprehensive atmospheric comparison figure...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# ----------------------------------------------------------------------------
# PANEL 1: Pressure Profile Comparison
# ----------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])

h_isa_agl = np.linspace(0, 1400, 300)
h_isa_asl = h_isa_agl + DUBENDORF_ELEV
T_isa_C = isa_temperature(h_isa_asl) - 273.15
p_isa = isa_pressure(h_isa_asl)

ax1.plot(p_isa, h_isa_agl, 'k--', lw=2, label='ISA', zorder=1)
ax1.plot(payerne['pressure'], payerne['height_agl'], 
         color='orange', lw=2.5, marker='o', markersize=3,
         label='Payerne (real)', zorder=2)
ax1.scatter(p_vamos_drop, h_vamos_drop, s=15, alpha=0.5, c='C0',
            label=f'VAMOS drop (N={len(p_vamos_drop)})', zorder=3)

ax1.set_xlabel('Pressure [hPa]', fontsize=11)
ax1.set_ylabel('Altitude [m AGL]', fontsize=11)
ax1.set_title('Pressure Profile\n(VAMOS matches Payerne → sensor OK)', 
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 1000)
ax1.invert_xaxis()

# Add text showing pressure agreement
p_payerne_ground = payerne.iloc[0]['pressure']
pressure_diff = p0_vamos - p_payerne_ground
ax1.text(0.05, 0.05, 
         f'Ground p:\nVAMOS: {p0_vamos:.1f} hPa\nPayerne: {p_payerne_ground:.1f} hPa\nΔ: {pressure_diff:+.1f} hPa',
         transform=ax1.transAxes,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
         fontsize=9, family='monospace', verticalalignment='bottom')

# ----------------------------------------------------------------------------
# PANEL 2: Air Density Profile
# ----------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])

# Compute densities
rho_isa = air_density(p_isa, isa_temperature(h_isa_asl))
rho_payerne = air_density(payerne['pressure'].values, payerne['temp'].values + 273.15)
rho_vamos = air_density(p_vamos_drop, T_vamos_drop + 273.15)

ax2.plot(rho_isa, h_isa_agl, 'k--', lw=2, label='ISA', zorder=1)
ax2.plot(rho_payerne, payerne['height_agl'], 
         color='orange', lw=2.5, marker='o', markersize=3,
         label='Payerne (real)', zorder=2)
ax2.scatter(rho_vamos, h_vamos_drop, s=15, alpha=0.5, c='C0',
            label='VAMOS drop', zorder=3)

ax2.set_xlabel('Air Density [kg/m³]', fontsize=11)
ax2.set_ylabel('Altitude [m AGL]', fontsize=11)
ax2.set_title('Air Density Profile\n(VAMOS underestimates ρ due to warm bias)', 
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 1000)

# Density difference text
rho_v_mean = np.mean(rho_vamos)
rho_p_mean = np.interp(np.mean(h_vamos_drop), payerne['height_agl'].values, rho_payerne)
rho_diff_pct = (rho_v_mean - rho_p_mean) / rho_p_mean * 100
ax2.text(0.05, 0.05,
         f'Mean ρ:\nVAMOS: {rho_v_mean:.3f}\nPayerne: {rho_p_mean:.3f}\nΔ: {rho_diff_pct:+.1f}%',
         transform=ax2.transAxes,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
         fontsize=9, family='monospace', verticalalignment='bottom')

# ----------------------------------------------------------------------------
# PANEL 3: Temperature with bias shown
# ----------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[0, 2])

ax3.plot(T_isa_C, h_isa_agl, 'k--', lw=2, label='ISA', zorder=1)
ax3.plot(payerne['temp'], payerne['height_agl'], 
         color='orange', lw=2.5, marker='o', markersize=3,
         label='Payerne (real)', zorder=2)
ax3.scatter(T_vamos_drop, h_vamos_drop, s=15, alpha=0.5, c='C0',
            label='VAMOS drop', zorder=3)

# Shade the bias region
T_payerne_interp = np.interp(h_vamos_drop, 
                              payerne['height_agl'].values[::-1],
                              payerne['temp'].values[::-1])
# Sort by altitude for fill_betweenx
sort_idx = np.argsort(h_vamos_drop)
ax3.fill_betweenx(h_vamos_drop[sort_idx], 
                   T_payerne_interp[sort_idx],
                   T_vamos_drop[sort_idx],
                   alpha=0.15, color='red', label='VAMOS warm bias')

ax3.set_xlabel('Temperature [°C]', fontsize=11)
ax3.set_ylabel('Altitude [m AGL]', fontsize=11)
ax3.set_title('Temperature Profile\n(VAMOS: large warm bias revealed)', 
              fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(alpha=0.3)
ax3.set_ylim(0, 1000)

# ----------------------------------------------------------------------------
# PANEL 4: SKEW-T DIAGRAM (Professional)
# ----------------------------------------------------------------------------
ax4 = fig.add_subplot(gs[1, 0])

# Create skew-T transformation
def skewT(T, p, skew=45):
    """Convert T,p to skewed coordinates."""
    Y = -np.log(p)  # log-pressure for y-axis
    X = T + skew * (-Y - np.log(1000))  # skew transformation
    return X, Y

# Define pressure range for skew-T
p_levels = np.arange(700, 1000, 10)
Y_levels = -np.log(p_levels)

# Draw background: dry adiabats (constant θ)
T_range = np.arange(-60, 50, 1)
theta_values = [250, 260, 270, 280, 290, 300, 310, 320]  # K
for theta in theta_values:
    T_line = theta * (p_levels / 1000) ** (Rd/Cp) - 273.15
    X_line, Y_line = skewT(T_line, p_levels)
    ax4.plot(X_line, Y_line, 'r-', lw=0.5, alpha=0.3)

# Draw background: isotherms (skewed)
for T_iso in np.arange(-40, 40, 10):
    T_line = np.full_like(p_levels, T_iso)
    X_line, Y_line = skewT(T_line, p_levels)
    ax4.plot(X_line, Y_line, 'b-', lw=0.5, alpha=0.3)

# Plot Payerne data
X_payerne_T, Y_payerne = skewT(payerne['temp'].values, payerne['pressure'].values)
X_payerne_Td, _ = skewT(payerne['dewpt'].values, payerne['pressure'].values)
ax4.plot(X_payerne_T, Y_payerne, 'r-', lw=2.5, label='Payerne T')
ax4.plot(X_payerne_Td, Y_payerne, 'g-', lw=2.5, label='Payerne Td (dewpoint)')

# Plot VAMOS data
X_vamos_T, Y_vamos = skewT(T_vamos_drop, p_vamos_drop)
ax4.plot(X_vamos_T, Y_vamos, 'bo', markersize=3, alpha=0.6, label='VAMOS T')

# Set axes
ax4.set_ylim(-np.log(1000), -np.log(750))  # ~750-1000 hPa
ax4.set_xlim(-30, 50)

# Y-axis ticks at standard pressures
y_ticks_p = [1000, 950, 925, 900, 850, 800, 775, 750]
ax4.set_yticks([-np.log(p) for p in y_ticks_p])
ax4.set_yticklabels([str(p) for p in y_ticks_p])

ax4.set_xlabel('Temperature [°C]  (isotherms skewed at 45°)', fontsize=10)
ax4.set_ylabel('Pressure [hPa]', fontsize=10)
ax4.set_title('Skew-T Log-P Diagram\n(red: dry adiabats, blue: isotherms)', 
              fontsize=12, fontweight='bold')
ax4.legend(fontsize=9, loc='upper left')
ax4.grid(alpha=0.2)

# ----------------------------------------------------------------------------
# PANEL 5: WIND HODOGRAPH (Payerne)
# ----------------------------------------------------------------------------
ax5 = fig.add_subplot(gs[1, 1])

# Plot Payerne wind hodograph
mask_hodo = payerne['height_agl'] <= 1000
u_payerne = payerne['u'].values[mask_hodo]
v_payerne = payerne['v'].values[mask_hodo]
h_payerne = payerne['height_agl'].values[mask_hodo]

# Color by altitude
sc = ax5.scatter(u_payerne, v_payerne, c=h_payerne, cmap='viridis', 
                  s=30, zorder=3, edgecolors='black', linewidths=0.5)
ax5.plot(u_payerne, v_payerne, '-', color='gray', lw=1, alpha=0.5, zorder=2)
plt.colorbar(sc, ax=ax5, label='Altitude [m AGL]')

# Plot VAMOS wind if available
if has_wind and wind_drop_mask.sum() > 10:
    # Decimate for clarity
    step = max(1, len(ux_drop) // 50)
    ax5.scatter(ux_drop[::step], uy_drop[::step], 
                c=h_wind_drop[::step], cmap='plasma',
                marker='^', s=30, alpha=0.5, label='VAMOS wind (drop)',
                zorder=4)

# Reference circles
for r in [2, 4, 6, 8, 10]:
    circle = plt.Circle((0, 0), r, fill=False, color='gray', 
                        linestyle=':', alpha=0.4)
    ax5.add_patch(circle)
    ax5.text(r*0.707, r*0.707, f'{r} m/s', fontsize=8, color='gray', alpha=0.6)

ax5.axhline(0, color='k', lw=0.5, alpha=0.3)
ax5.axvline(0, color='k', lw=0.5, alpha=0.3)
ax5.set_xlabel('U (East-West) [m/s]', fontsize=11)
ax5.set_ylabel('V (North-South) [m/s]', fontsize=11)
ax5.set_title('Wind Hodograph\n(Payerne: color = altitude)', 
              fontsize=12, fontweight='bold')
ax5.set_aspect('equal')
ax5.grid(alpha=0.3)
ax5.set_xlim(-12, 12)
ax5.set_ylim(-12, 12)

# ----------------------------------------------------------------------------
# PANEL 6: VERTICAL WIND SHEAR
# ----------------------------------------------------------------------------
ax6 = fig.add_subplot(gs[1, 2])

# Compute vertical wind shear from Payerne
h_p = payerne['height_agl'].values
u_p = payerne['u'].values
v_p = payerne['v'].values
ws_p = payerne['wind_speed'].values

# Wind shear magnitude: d|V|/dz
# Use smoothed values to reduce noise
from scipy.ndimage import uniform_filter1d
u_smooth = uniform_filter1d(u_p, size=5)
v_smooth = uniform_filter1d(v_p, size=5)

# Shear components
du_dz = np.gradient(u_smooth, h_p)
dv_dz = np.gradient(v_smooth, h_p)
shear_mag = np.sqrt(du_dz**2 + dv_dz**2) * 1000  # convert to per km

# Only plot where altitude increases monotonically
valid = h_p <= 1000

ax6.plot(shear_mag[valid], h_p[valid], 'o-', color='purple', lw=2, 
         markersize=4, label='Payerne shear')
ax6.fill_betweenx(h_p[valid], 0, shear_mag[valid], alpha=0.2, color='purple')

# Wind speed as reference (right axis)
ax6_r = ax6.twiny()
ax6_r.plot(ws_p[valid], h_p[valid], 's-', color='darkgreen', lw=1.5, 
           markersize=4, alpha=0.7, label='Wind speed')
ax6_r.set_xlabel('Wind Speed [m/s]', fontsize=11, color='darkgreen')
ax6_r.tick_params(axis='x', labelcolor='darkgreen')

ax6.set_xlabel('Vertical Wind Shear [m/s per km]', fontsize=11, color='purple')
ax6.tick_params(axis='x', labelcolor='purple')
ax6.set_ylabel('Altitude [m AGL]', fontsize=11)
ax6.set_title('Vertical Wind Shear Profile\n(Payerne radiosonde)', 
              fontsize=12, fontweight='bold')
ax6.grid(alpha=0.3)
ax6.set_ylim(0, 1000)
ax6.set_xlim(0, max(shear_mag[valid]) * 1.1)

# Annotate max shear layer
max_shear_idx = np.argmax(shear_mag[valid])
max_shear_h = h_p[valid][max_shear_idx]
max_shear_val = shear_mag[valid][max_shear_idx]
ax6.annotate(f'Max shear:\n{max_shear_val:.1f} m/s/km\nat {max_shear_h:.0f} m',
             xy=(max_shear_val, max_shear_h),
             xytext=(max_shear_val*0.5, max_shear_h+200),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ----------------------------------------------------------------------------
# PANEL 7: VAMOS WIND HODOGRAPH (if data available)
# ----------------------------------------------------------------------------
if has_wind and 'ux_drop' in locals() and len(ux_drop) > 10:
    ax7 = fig.add_subplot(gs[2, 0])
    
    # Decimate for clarity
    step = max(1, len(ux_drop) // 100)
    u_vamos_plot = ux_drop[::step]
    v_vamos_plot = uy_drop[::step]
    h_vamos_plot = h_wind_drop[::step]
    
    # Scatter plot colored by altitude
    sc7 = ax7.scatter(u_vamos_plot, v_vamos_plot, c=h_vamos_plot, cmap='plasma',
                      s=30, zorder=3, edgecolors='black', linewidths=0.5,
                      vmin=0, vmax=h_vamos_plot.max())
    ax7.plot(u_vamos_plot, v_vamos_plot, '-', color='gray', lw=1, alpha=0.3, zorder=2)
    
    # Colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider7 = make_axes_locatable(ax7)
    cax7 = divider7.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(sc7, cax=cax7, label='Altitude [m]')
    
    # Reference circles
    for r in [1, 2, 3, 4, 5]:
        circle = plt.Circle((0, 0), r, fill=False, color='gray', 
                           linestyle=':', alpha=0.4)
        ax7.add_patch(circle)
    
    # Mark start/end
    ax7.scatter(u_vamos_plot[0], v_vamos_plot[0], s=100, marker='o',
               color='green', edgecolor='darkgreen', linewidth=2,
               zorder=5, label=f'Start ({h_vamos_plot[0]:.0f} m)')
    ax7.scatter(u_vamos_plot[-1], v_vamos_plot[-1], s=100, marker='s',
               color='red', edgecolor='darkred', linewidth=2,
               zorder=5, label=f'End ({h_vamos_plot[-1]:.0f} m)')
    
    ax7.axhline(0, color='k', lw=0.5, alpha=0.3)
    ax7.axvline(0, color='k', lw=0.5, alpha=0.3)
    ax7.set_xlabel('U (East) [m/s]', fontsize=11)
    ax7.set_ylabel('V (North) [m/s]', fontsize=11)
    ax7.set_title('VAMOS Wind Hodograph\n(scatter shows rotation/oscillation)', 
                  fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9, loc='upper left')
    ax7.set_aspect('equal')
    ax7.grid(alpha=0.3)
    
    lim7 = max(np.abs(u_vamos_plot).max(), np.abs(v_vamos_plot).max()) * 1.2
    ax7.set_xlim(-lim7, lim7)
    ax7.set_ylim(-lim7, lim7)
    
    # Mean wind stats
    mean_u_v = np.mean(ux_drop)
    mean_v_v = np.mean(uy_drop)
    mean_ws_v = np.sqrt(mean_u_v**2 + mean_v_v**2)
    ax7.text(0.02, 0.02, f'Mean: {mean_ws_v:.1f} m/s',
             transform=ax7.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('VAMOS CanSat vs Payerne Radiosonde — Complete Atmospheric Analysis (6 Feb 2026)',
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig(f"{OUTPUT_DIR}/fig8_complete_atmospheric.png", dpi=130, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: {OUTPUT_DIR}/fig8_complete_atmospheric.png")

# ============================================================================
# FIGURE 2: VAMOS Wind Hodograph (Standalone)
# ============================================================================

if has_wind and wind_drop_mask.sum() > 10:
    print("\nGenerating VAMOS-only hodograph...")
    
    fig_hodo, ax_hodo = plt.subplots(figsize=(10, 10))
    
    # Decimate for clarity
    step = max(1, len(ux_drop) // 100)
    u_plot = ux_drop[::step]
    v_plot = uy_drop[::step]
    h_plot = h_wind_drop[::step]
    
    # Main scatter plot
    sc = ax_hodo.scatter(u_plot, v_plot, c=h_plot, cmap='viridis', 
                        s=40, zorder=3, edgecolors='black', linewidths=0.5,
                        vmin=0, vmax=h_plot.max())
    
    # Connect with line
    ax_hodo.plot(u_plot, v_plot, '-', color='gray', lw=1, alpha=0.3, zorder=2)
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax_hodo, label='Altitude [m AGL]', pad=0.02)
    cbar.ax.tick_params(labelsize=11)
    
    # Reference circles
    max_wind = max(np.abs(u_plot).max(), np.abs(v_plot).max()) * 1.1
    for speed in [1, 2, 3, 4, 5]:
        if speed < max_wind:
            from matplotlib.patches import Circle
            circle = Circle((0, 0), speed, fill=False, color='gray', 
                           linestyle=':', alpha=0.4, linewidth=1)
            ax_hodo.add_patch(circle)
            label_x = speed * np.cos(np.pi/4)
            label_y = speed * np.sin(np.pi/4)
            ax_hodo.text(label_x, label_y, f'{speed} m/s', 
                   fontsize=9, color='gray', alpha=0.6,
                   ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.7, edgecolor='none'))
    
    # Mark start/end
    ax_hodo.scatter(u_plot[0], v_plot[0], s=200, marker='o', 
              color='green', edgecolor='darkgreen', linewidth=2.5,
              zorder=5, label=f'Apogee (h={h_plot[0]:.0f} m)')
    ax_hodo.scatter(u_plot[-1], v_plot[-1], s=200, marker='s', 
              color='red', edgecolor='darkred', linewidth=2.5,
              zorder=5, label=f'Landing (h={h_plot[-1]:.0f} m)')
    
    # Axes
    ax_hodo.axhline(0, color='k', linewidth=0.8, alpha=0.3)
    ax_hodo.axvline(0, color='k', linewidth=0.8, alpha=0.3)
    ax_hodo.set_xlabel('U (Eastward Wind) [m/s]', fontsize=13, fontweight='bold')
    ax_hodo.set_ylabel('V (Northward Wind) [m/s]', fontsize=13, fontweight='bold')
    ax_hodo.set_title(f'VAMOS Wind Hodograph During Drop\n'
                 f'(t={t_apogee:.0f}-{t_landing:.0f} s, N={len(u_plot)} decimated samples, fs≈1 Hz)',
                 fontsize=14, fontweight='bold', pad=15)
    ax_hodo.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax_hodo.set_aspect('equal')
    ax_hodo.grid(True, alpha=0.3)
    
    # Symmetric limits
    lim = max(np.abs(u_plot).max(), np.abs(v_plot).max()) * 1.2
    ax_hodo.set_xlim(-lim, lim)
    ax_hodo.set_ylim(-lim, lim)
    
    # Cardinal directions
    text_offset = lim * 0.93
    ax_hodo.text(text_offset, 0, 'E', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='darkblue')
    ax_hodo.text(-text_offset, 0, 'W', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='darkblue')
    ax_hodo.text(0, text_offset, 'N', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='darkblue')
    ax_hodo.text(0, -text_offset, 'S', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='darkblue')
    
    # Statistics
    mean_u = np.mean(ux_drop)
    mean_v = np.mean(uy_drop)
    mean_speed = np.sqrt(mean_u**2 + mean_v**2)
    mean_dir = np.rad2deg(np.arctan2(-mean_u, -mean_v)) % 360
    
    stats_text = (f'Mean wind during drop:\n'
                  f'  Speed: {mean_speed:.2f} m/s\n'
                  f'  From: {mean_dir:.0f}°\n'
                  f'  U: {mean_u:+.2f} m/s\n'
                  f'  V: {mean_v:+.2f} m/s\n\n'
                  f'Variability (scatter):\n'
                  f'  σ(U): {np.std(ux_drop):.2f} m/s\n'
                  f'  σ(V): {np.std(uy_drop):.2f} m/s')
    
    ax_hodo.text(0.98, 0.02, stats_text,
            transform=ax_hodo.transAxes,
            fontsize=10, family='monospace',
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Add explanation note
    note_text = ('Note: Scatter reflects CanSat rotation/oscillation\n'
                 'under parachute. Compare with Payerne hodograph\n'
                 'which shows smooth vertical profile from stable balloon.')
    ax_hodo.text(0.02, 0.98, note_text,
            transform=ax_hodo.transAxes,
            fontsize=9, style='italic',
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig9_vamos_hodograph.png", dpi=130, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {OUTPUT_DIR}/fig9_vamos_hodograph.png")
else:
    print("\n  ⚠️  Not enough VAMOS wind data during drop to create hodograph")

# ============================================================================
# PRINT SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

# Pressure agreement
print(f"\n[PRESSURE] VAMOS vs Payerne:")
print(f"  Ground pressure VAMOS: {p0_vamos:.2f} hPa")
print(f"  Ground pressure Payerne: {p_payerne_ground:.2f} hPa")
print(f"  Difference: {pressure_diff:+.2f} hPa (~{pressure_diff*8:.0f} m altitude equiv)")

# Density
print(f"\n[DENSITY] At {np.mean(h_vamos_drop):.0f} m AGL mean:")
print(f"  VAMOS ρ:   {rho_v_mean:.4f} kg/m³")
print(f"  Payerne ρ: {rho_p_mean:.4f} kg/m³")
print(f"  Bias: {rho_diff_pct:+.2f}% (VAMOS underestimates due to warm T)")

# Wind statistics
print(f"\n[WIND] Payerne:")
print(f"  Surface wind: {payerne.iloc[0]['wind_speed']:.1f} m/s at {payerne.iloc[0]['wind_dir']:.0f}°")
print(f"  Max wind in layer: {ws_p[valid].max():.1f} m/s")
print(f"  Max vertical shear: {max_shear_val:.1f} m/s per km at {max_shear_h:.0f} m")
print(f"  Wind turning (surface to 1km): ~{payerne['wind_dir'].values[-1] - payerne['wind_dir'].values[0]:.0f}°")

# Save summary
summary = {
    "pressure": {
        "vamos_ground_hPa": float(p0_vamos),
        "payerne_ground_hPa": float(p_payerne_ground),
        "difference_hPa": float(pressure_diff)
    },
    "density": {
        "vamos_mean": float(rho_v_mean),
        "payerne_mean": float(rho_p_mean),
        "bias_percent": float(rho_diff_pct)
    },
    "wind": {
        "surface_speed_mps": float(payerne.iloc[0]['wind_speed']),
        "surface_dir_deg": float(payerne.iloc[0]['wind_dir']),
        "max_speed_mps": float(ws_p[valid].max()),
        "max_shear_per_km": float(max_shear_val),
        "max_shear_altitude_m": float(max_shear_h)
    }
}

with open(f"{OUTPUT_DIR}/atmospheric_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Saved: {OUTPUT_DIR}/atmospheric_summary.json")
print("="*70)

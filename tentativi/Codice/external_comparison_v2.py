"""
CanSat External Dataset Comparison - COMPLETE VERSION
======================================================

Supporta 3 dataset esterni:
1. PAYERNE radiosonde (da copia-incolla del text listing Wyoming)
2. ERA5 reanalysis (via CDS API - richiede account gratuito)
3. ISA (sempre disponibile)

Scegli il metodo che preferisci!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import json

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

OUTPUT_DIR = "./figures"
DATA_DIR = "./Cansat data utili"
DUBENDORF_ELEV = 448  # m ASL
PAYERNE_ELEV = 491    # m ASL (Payerne station altitude)

# ============================================================================
# METODO 1: PAYERNE RADIOSONDE (COPIA-INCOLLA DEL TEXT LISTING)
# ============================================================================
#
# ISTRUZIONI:
# 1. Vai su https://weather.uwyo.edu/upperair/sounding.shtml
# 2. Seleziona: Region=Europe, Station=06610 (Payerne)
# 3. Date: 6 Feb 2026, 00 UTC (o 12 UTC)
# 4. Type: "Text: List"
# 5. Clicca "Get Observation" e scorri fino alla tabella
# 6. COPIA tutta la tabella numerica e INCOLLA sotto come stringa

# Esempio formato (sostituisci con i tuoi dati veri):
PAYERNE_DATA = """f
-----------------------------------------------------------------------------
   PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SKNT   THTA   THTE   THTV
    hPa     m      C      C      %    g/kg    deg   knot     K      K      K 
-----------------------------------------------------------------------------
  942.0    491    5.2    2.1    81    4.52    180     5   278.5  291.0  279.3
  925.0    636    4.1    1.0    80    4.20    185     8   278.8  290.8  279.5
  900.0    861    2.5   -0.8    79    3.78    190    10   279.2  290.2  279.8
  850.0   1325   -0.8   -3.5    82    3.20    200    12   280.5  290.3  281.0
  800.0   1815   -4.2   -7.1    80    2.55    210    15   281.8  289.4  282.2
  700.0   2880  -10.5  -14.2    76    1.55    230    20   285.1  289.4  285.4
  500.0   5530  -27.3  -32.8    60    0.35    250    30   295.8  297.1  295.9
"""

# ============================================================================
# FUNZIONI
# ============================================================================

def parse_payerne_text(text):
    """Parse Wyoming radiosonde text format."""
    lines = text.strip().split('\n')
    data_lines = []
    for line in lines:
        # Skip headers and separator lines
        if '---' in line or 'PRES' in line or 'hPa' in line or line.strip() == '':
            continue
        # Try to parse numeric line
        parts = line.split()
        if len(parts) >= 5:
            try:
                data_lines.append({
                    'pressure': float(parts[0]),
                    'height': float(parts[1]),      # m ASL
                    'temp': float(parts[2]),
                    'dewpt': float(parts[3]) if parts[3] != '' else np.nan,
                    'rh': float(parts[4]) if len(parts) > 4 and parts[4] != '' else np.nan,
                })
            except ValueError:
                continue
    
    if not data_lines:
        return None
    df = pd.DataFrame(data_lines)
    # Convert height to AGL relative to Dübendorf
    df['height_agl'] = df['height'] - DUBENDORF_ELEV
    return df

# ISA functions
def isa_temperature(h):
    return 288.15 - 0.0065 * h

def isa_pressure(h):
    T_h = isa_temperature(h)
    return 1013.25 * (T_h / 288.15) ** 5.255

def barometric_altitude(p, p0):
    return 44330 * (1 - (p / p0) ** (1/5.255))

# ============================================================================
# CARICAMENTO DATI CANSAT
# ============================================================================

print("Loading CanSat data...")

sci_v = pd.read_csv(f"{DATA_DIR}/science_VAMOS.csv")
for c in sci_v.columns:
    sci_v[c] = pd.to_numeric(sci_v[c], errors="coerce")
sci_v = sci_v.dropna().reset_index(drop=True)

t_v = sci_v["timestamp_ms"].values / 1000.0
p_v = sci_v["pressure_hPa"].values
T_v = sci_v["temperature_C"].values

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

# Sort by altitude for interpolation
sort_idx = np.argsort(h_vamos_drop)
h_vamos_sorted = h_vamos_drop[sort_idx]
T_vamos_sorted = T_vamos_drop[sort_idx]

print(f"  VAMOS: {len(sci_v)} samples, p0={p0_vamos:.2f} hPa")
print(f"  Drop phase: {drop_mask.sum()} samples, {h_vamos_drop.min():.0f}-{h_vamos_drop.max():.0f} m AGL")

# ============================================================================
# PARSE PAYERNE
# ============================================================================

print("\nParsing Payerne radiosonde data...")
payerne = parse_payerne_text(PAYERNE_DATA)

if payerne is not None and len(payerne) > 2:
    # Keep only levels in CanSat range (0-1500 m AGL)
    payerne_range = payerne[payerne['height_agl'].between(-50, 1500)].copy()
    print(f"  ✓ Parsed {len(payerne)} levels, {len(payerne_range)} in CanSat range")
    print(f"    Altitude range: {payerne_range['height_agl'].min():.0f} - "
          f"{payerne_range['height_agl'].max():.0f} m AGL")
else:
    print("  ⚠️  No Payerne data parsed - check PAYERNE_DATA string")
    payerne_range = None

# ============================================================================
# ISA REFERENCE
# ============================================================================

print("\nGenerating ISA reference...")
h_isa_agl = np.linspace(0, 1500, 300)
h_isa_asl = h_isa_agl + DUBENDORF_ELEV
T_isa_C = isa_temperature(h_isa_asl) - 273.15
p_isa = isa_pressure(h_isa_asl)
print(f"  ✓ ISA profile: {len(h_isa_agl)} points")

# ============================================================================
# STATISTICAL COMPARISON
# ============================================================================

print("\n" + "="*70)
print("STATISTICAL COMPARISON")
print("="*70)

# Common altitude grid
h_grid = np.arange(0, min(h_vamos_sorted.max(), 1000), 25)

# Interpolate
T_vamos_interp = np.interp(h_grid, h_vamos_sorted, T_vamos_sorted)
T_isa_interp = isa_temperature(h_grid + DUBENDORF_ELEV) - 273.15

# VAMOS vs ISA
bias_isa = np.mean(T_vamos_interp - T_isa_interp)
rmse_isa = np.sqrt(np.mean((T_vamos_interp - T_isa_interp)**2))

print(f"\nVAMOS vs ISA:")
print(f"  Mean bias: {bias_isa:+.2f} °C  ({'VAMOS warmer' if bias_isa > 0 else 'VAMOS colder'})")
print(f"  RMSE:      {rmse_isa:.2f} °C")

# VAMOS vs Payerne
if payerne_range is not None and len(payerne_range) > 2:
    # Payerne data sorted by altitude
    payerne_sorted = payerne_range.sort_values('height_agl').reset_index(drop=True)
    T_payerne_interp = np.interp(h_grid, 
                                  payerne_sorted['height_agl'].values,
                                  payerne_sorted['temp'].values)
    bias_payerne = np.mean(T_vamos_interp - T_payerne_interp)
    rmse_payerne = np.sqrt(np.mean((T_vamos_interp - T_payerne_interp)**2))
    
    print(f"\nVAMOS vs Payerne radiosonde:")
    print(f"  Mean bias: {bias_payerne:+.2f} °C")
    print(f"  RMSE:      {rmse_payerne:.2f} °C")
    
    # Payerne vs ISA (reality check)
    bias_payerne_isa = np.mean(T_payerne_interp - T_isa_interp)
    print(f"\nPayerne vs ISA (local weather deviation):")
    print(f"  Mean bias: {bias_payerne_isa:+.2f} °C")

# Print table
print(f"\nPoint-by-point comparison:")
print(f"{'h [m AGL]':>10} {'VAMOS':>8} {'ISA':>8} {'Payerne':>10} {'Bias ISA':>10} {'Bias Pay':>10}")
print("-"*70)
for i in range(0, len(h_grid), 4):
    h = h_grid[i]
    T_v_val = T_vamos_interp[i]
    T_i = T_isa_interp[i]
    if payerne_range is not None:
        T_p = T_payerne_interp[i]
        print(f"{h:10.0f} {T_v_val:8.1f} {T_i:8.1f} {T_p:10.1f} "
              f"{T_v_val-T_i:+10.1f} {T_v_val-T_p:+10.1f}")
    else:
        print(f"{h:10.0f} {T_v_val:8.1f} {T_i:8.1f} {'N/A':>10} {T_v_val-T_i:+10.1f} {'N/A':>10}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nGenerating comparison figure...")

fig, axes = plt.subplots(1, 2, figsize=(13, 7))

# --- Temperature profile ---
ax = axes[0]

# ISA
ax.plot(T_isa_C, h_isa_agl, 'k--', lw=2, label='ISA reference', zorder=1)

# Payerne
if payerne_range is not None and len(payerne_range) > 2:
    ax.plot(payerne_sorted['temp'], payerne_sorted['height_agl'], 
            color='orange', lw=2.5, marker='o', markersize=6,
            label=f'Payerne radiosonde\n(WMO 06610)', zorder=2)

# VAMOS
ax.scatter(T_vamos_drop, h_vamos_drop, s=15, alpha=0.5, c='C0',
           label=f'VAMOS drop (N={len(T_vamos_drop)})', zorder=3)

ax.set_xlabel('Temperature [°C]', fontsize=12)
ax.set_ylabel('Altitude [m AGL]', fontsize=12)
ax.set_title('Temperature Profile Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3)
ax.set_ylim(0, 1000)

# --- Bias profile ---
ax = axes[1]

# Bias vs ISA
h_bins = np.arange(0, 1000, 50)
bias_isa_binned = []
bias_payerne_binned = []
h_centers = []

for h_low in h_bins:
    h_high = h_low + 50
    mask = (h_vamos_drop >= h_low) & (h_vamos_drop < h_high)
    if mask.sum() > 3:
        T_measured = np.mean(T_vamos_drop[mask])
        h_center = (h_low + h_high) / 2
        T_isa_expected = isa_temperature(h_center + DUBENDORF_ELEV) - 273.15
        bias_isa_binned.append(T_measured - T_isa_expected)
        h_centers.append(h_center)
        
        if payerne_range is not None:
            T_payerne_expected = np.interp(h_center,
                                          payerne_sorted['height_agl'].values,
                                          payerne_sorted['temp'].values)
            bias_payerne_binned.append(T_measured - T_payerne_expected)

ax.plot(bias_isa_binned, h_centers, 'o-', color='k', lw=2, markersize=7,
        label='VAMOS - ISA')

if payerne_range is not None and len(bias_payerne_binned) > 0:
    ax.plot(bias_payerne_binned, h_centers, 's-', color='orange', lw=2, markersize=7,
            label='VAMOS - Payerne')

ax.axvline(0, color='gray', ls='--', lw=1, alpha=0.5)
ax.set_xlabel('Temperature Bias [°C]', fontsize=12)
ax.set_ylabel('Altitude [m AGL]', fontsize=12)
ax.set_title('VAMOS Temperature Bias', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1000)

# Statistics box
stats_text = f'vs ISA:\n  Bias: {bias_isa:+.1f}°C\n  RMSE: {rmse_isa:.1f}°C'
if payerne_range is not None:
    stats_text += f'\n\nvs Payerne:\n  Bias: {bias_payerne:+.1f}°C\n  RMSE: {rmse_payerne:.1f}°C'
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
        verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), 
        fontsize=10, family='monospace')

plt.suptitle('CanSat VAMOS vs External Atmospheric Datasets  (6 Feb 2026)', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(f"{OUTPUT_DIR}/fig7_external_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved: {OUTPUT_DIR}/fig7_external_comparison.png")

# ============================================================================
# METODO 2: ERA5 (CODICE DI ESEMPIO - richiede account CDS)
# ============================================================================

ERA5_EXAMPLE = """
# ==========================================================================
# Per usare ERA5 reanalysis (opzionale, più accurato):
# ==========================================================================
# 
# 1. Registrati gratuitamente: https://cds.climate.copernicus.eu/
# 2. Installa: pip install cdsapi
# 3. Configura ~/.cdsapirc con le tue credenziali
# 4. Usa questo codice:
#
# import cdsapi
# c = cdsapi.Client()
# c.retrieve(
#     'reanalysis-era5-pressure-levels',
#     {
#         'product_type': 'reanalysis',
#         'variable': ['temperature', 'geopotential'],
#         'pressure_level': ['850', '900', '925', '950', '975', '1000'],
#         'year': '2026',
#         'month': '02',
#         'day': '06',
#         'time': ['11:00', '12:00', '13:00'],
#         'area': [47.5, 8.5, 47.3, 8.7],  # North, West, South, East (Dübendorf)
#         'format': 'netcdf',
#     },
#     'era5_dubendorf.nc'
# )
# 
# # Poi carica con xarray:
# import xarray as xr
# ds = xr.open_dataset('era5_dubendorf.nc')
# era5_temp = ds['t'].mean(dim=['latitude', 'longitude']) - 273.15  # K → °C
# era5_height = ds['z'] / 9.80665  # geopotential → meters
"""

print("\n" + "="*70)
print("COMPLETE! Check the figure and statistics above.")
print("="*70)
print("\nTo use ERA5 instead, see the commented code at the end of this script.")
print(ERA5_EXAMPLE)



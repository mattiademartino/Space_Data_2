
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import datetime

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
FILE = '/Users/alice/Desktop/Space_Data_2/alice/ERA5/era5_file.nc'
ds = nc.Dataset(FILE)

pressure_levels = ds.variables['pressure_level'][:]   # hPa,  shape (9,)
temp_K          = ds.variables['t'][:]                 # K,    shape (24, 9, lat, lon)
humidity        = ds.variables['r'][:]                 # %,    shape (24, 9, lat, lon)
geopot          = ds.variables['z'][:]                 # m²/s², shape (24, 9, lat, lon)
valid_times     = ds.variables['valid_time'][:]        # seconds since 1970-01-01

# Reference date string for plot title
t0 = datetime.datetime.fromtimestamp(int(valid_times[0]),  tz=datetime.timezone.utc)
t1 = datetime.datetime.fromtimestamp(int(valid_times[-1]), tz=datetime.timezone.utc)
date_str = f"{t0.strftime('%d %b %Y')}  ({t0.strftime('%H:%M')} – {t1.strftime('%H:%M')} UTC)"

# ─────────────────────────────────────────────
# 2. COMPUTE MEAN VERTICAL PROFILES
#    Average over time (axis 0), latitude (axis 2), longitude (axis 3)
# ─────────────────────────────────────────────
axis = (0, 2, 3)

altitude = np.mean(geopot,   axis=axis) / 9.80665   # geopotential height [m]
temp_C   = np.mean(temp_K,   axis=axis) - 273.15    # temperature [°C]
rh       = np.mean(humidity, axis=axis)              # relative humidity [%]

# Use all levels (no altitude filter — data already spans 1000–800 hPa)
altitude_f = altitude
temp_f     = temp_C
rh_f       = rh
pres_f     = pressure_levels

# ─────────────────────────────────────────────
# 3. PRINT DATA TABLE
# ─────────────────────────────────────────────
header = f"{'P [hPa]':>10} {'Altitude [m]':>14} {'Temperature [°C]':>18} {'Rel. Humidity [%]':>18}"
sep    = "=" * len(header)
print(f"\n{sep}")
print("ERA5 – Mean Vertical Profile (all levels, 1000–800 hPa)")
print(date_str)
print(sep)
print(header)
print("-" * len(header))
for i in range(len(altitude_f)):
    print(f"{pres_f[i]:>10.0f} {altitude_f[i]:>14.1f} {temp_f[i]:>18.2f} {rh_f[i]:>18.1f}")
print(sep + "\n")

# ─────────────────────────────────────────────
# 4. PLOT
# ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
fig.suptitle(
    f'ERA5 – Vertical Profiles (1000–800 hPa)\n{date_str}',
    fontsize=13, fontweight='bold'
)

alt_max = altitude_f.max() * 1.05   # 5% padding above top level

# ── Temperature ──
ax1.plot(temp_f, altitude_f,
         'o-', color='#E84040', linewidth=2.5,
         markersize=8, markerfacecolor='white',
         markeredgewidth=2.5, markeredgecolor='#E84040')

for t, a, p in zip(temp_f, altitude_f, pres_f):
    ax1.annotate(f'{p:.0f} hPa', xy=(t, a),
                 xytext=(7, 2), textcoords='offset points',
                 fontsize=9, color='#555555')

ax1.set_xlabel('Temperature [°C]', fontsize=11)
ax1.set_ylabel('Geopotential Height [m]', fontsize=11)
ax1.set_title('Temperature', fontsize=12, fontweight='bold', color='#E84040')
ax1.set_ylim(0, alt_max)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_facecolor('white')

# ── Relative Humidity ──
ax2.plot(rh_f, altitude_f,
         'o-', color='#3B82F6', linewidth=2.5,
         markersize=8, markerfacecolor='white',
         markeredgewidth=2.5, markeredgecolor='#3B82F6')

for r, a, p in zip(rh_f, altitude_f, pres_f):
    ax2.annotate(f'{p:.0f} hPa', xy=(r, a),
                 xytext=(7, 2), textcoords='offset points',
                 fontsize=9, color='#555555')

ax2.set_xlabel('Relative Humidity [%]', fontsize=11)
ax2.set_title('Relative Humidity', fontsize=12, fontweight='bold', color='#3B82F6')
ax2.set_ylim(0, alt_max)
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.set_facecolor('white')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("/Users/alice/Desktop/Space_Data_2/alice/ERA5/wRA5_T_RH_profiles.png",
            dpi=150, bbox_inches="tight")
plt.show()

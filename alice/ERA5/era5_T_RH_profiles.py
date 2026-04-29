import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

out_dir = "/Users/alice/Desktop/Space_Data_2/alice/ERA5"
os.makedirs(out_dir, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
FILE = '/Users/alice/Desktop/Space_Data_2/alice/ERA5/era5_file.nc'   # cambia con il tuo path locale se necessario
ds = xr.open_dataset(FILE)

# ─────────────────────────────────────────────
# 2. SELECT TIME: 5 FEBRUARY 2026 AT 15:00 UTC
# ─────────────────────────────────────────────
target = np.datetime64("2026-02-05T15:00:00")

ds_time = ds.sel(valid_time=target, method="nearest")

selected_time = pd.to_datetime(ds_time.valid_time.values)
date_str = selected_time.strftime("%d %b %Y  %H:%M UTC")

print(f"Selected time step: {date_str}")

# ─────────────────────────────────────────────
# 3. FILTER PRESSURE LEVELS: 950 hPa → 800 hPa
# ─────────────────────────────────────────────
ds_prof = ds_time.sel(
    pressure_level=slice(950, 800)
)

# Se i livelli sono ordinati al contrario, usa questo fallback
ds_prof = ds_time.where(
    (ds_time.pressure_level >= 800) &
    (ds_time.pressure_level <= 950),
    drop=True
)

pres_f = ds_prof.pressure_level.values

# ─────────────────────────────────────────────
# 4. AREA-MEAN VERTICAL PROFILE
# ─────────────────────────────────────────────
temp_C = ds_prof["t"].mean(dim=("latitude", "longitude")).values - 273.15
rh     = ds_prof["r"].mean(dim=("latitude", "longitude")).values

# ─────────────────────────────────────────────
# 5. PRINT DATA TABLE
# ─────────────────────────────────────────────
header = f"{'P [hPa]':>10} {'Temperature [°C]':>18} {'Rel. Humidity [%]':>18}"
sep = "=" * len(header)

print(f"\n{sep}")
print("ERA5 – Vertical Profile (950–800 hPa)")
print(date_str)
print(sep)
print(header)
print("-" * len(header))

for p, t, h in zip(pres_f, temp_C, rh):
    print(f"{p:>10.0f} {t:>18.2f} {h:>18.1f}")

print(sep + "\n")

# ─────────────────────────────────────────────
# 6. PLOT
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))

fig.suptitle(
    f"ERA5 – Temperature Profile (950–800 hPa)\n{date_str}",
    fontsize=13,
    fontweight="bold"
)

# Temperature
ax.plot(
    temp_C, pres_f,
    "o-",
    color="#E84040",
    linewidth=2.5,
    markersize=8,
    markerfacecolor="white",
    markeredgewidth=2.5,
    markeredgecolor="#E84040"
)

ax.set_xlabel("Temperature [°C]", fontsize=11)
ax.set_ylabel("Pressure [hPa]", fontsize=11)
ax.set_title("Temperature", fontsize=12, fontweight="bold", color="#E84040")

ax.grid(True, linestyle="--", alpha=0.4)
ax.set_facecolor("white")

# pressione alta in basso, pressione bassa in alto
ax.invert_yaxis()

plt.tight_layout(rect=[0, 0, 1, 0.93])

out_file = os.path.join(out_dir, "ERA5_T_profile_20260205_15UTC.png")

plt.savefig(
    out_file,
    dpi=150,
    bbox_inches="tight"
)

print(f"Figura salvata in: {out_file}")

plt.show()
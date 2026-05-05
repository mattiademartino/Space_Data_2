import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ─────────────────────────────────────────────
# PATH
# ─────────────────────────────────────────────
sma_file = "/Users/alice/Desktop/Space_Data_2/alice/Meteoswiss/ogd-smn_sma_h_recent.csv"
ueb_file = "/Users/alice/Desktop/Space_Data_2/alice/Meteoswiss/ogd-smn-tower_ueb_h_recent.csv"

out_dir = "/Users/alice/Desktop/Space_Data_2/alice/Meteoswiss/img"
os.makedirs(out_dir, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
sma = pd.read_csv(
    sma_file,
    sep=";",
    parse_dates=["reference_timestamp"],
    dayfirst=True,
    low_memory=False
)

ueb = pd.read_csv(
    ueb_file,
    sep=";",
    parse_dates=["reference_timestamp"],
    dayfirst=True,
    low_memory=False
)

sma.columns = sma.columns.str.strip()
ueb.columns = ueb.columns.str.strip()

# ─────────────────────────────────────────────
# SELECT 5 FEBRUARY 2026
# ─────────────────────────────────────────────
date = pd.Timestamp("2026-02-05").date()

sma_day = sma[sma["reference_timestamp"].dt.date == date].copy()
ueb_day = ueb[ueb["reference_timestamp"].dt.date == date].copy()

print("Righe SMA:", len(sma_day))
print("Righe UEB:", len(ueb_day))

# ─────────────────────────────────────────────
# CONVERT NUMERIC COLUMNS
# ─────────────────────────────────────────────
sma_cols = ["tre200h0", "ure200h0", "prestah0", "fu3010h0", "fu3010h1"]
ueb_cols = ["ta1towh0"]

for col in sma_cols:
    sma_day[col] = pd.to_numeric(sma_day[col], errors="coerce")

for col in ueb_cols:
    ueb_day[col] = pd.to_numeric(ueb_day[col], errors="coerce")

# ─────────────────────────────────────────────
# FIGURE 1: METEOROLOGICAL CONDITIONS SMA
# ─────────────────────────────────────────────
sma_day["wind_kts"] = sma_day["fu3010h0"] * 1.944
sma_day["gust_kts"] = sma_day["fu3010h1"] * 1.944

t = sma_day["reference_timestamp"]

fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

fig.suptitle(
    "Meteorological Conditions – Zürich/Fluntern (SMA)\n5 February 2026",
    fontsize=14,
    fontweight="bold"
)

# Temperature
axes[0].plot(t, sma_day["tre200h0"], color="tomato", linewidth=2)
axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
axes[0].fill_between(
    t,
    sma_day["tre200h0"],
    0,
    where=sma_day["tre200h0"] < 0,
    alpha=0.12,
    color="blue"
)
axes[0].set_ylabel("Temperature (°C)")
axes[0].grid(True, alpha=0.3)

# Wind
axes[1].plot(t, sma_day["wind_kts"], color="steelblue", linewidth=2, label="Mean wind")
axes[1].plot(t, sma_day["gust_kts"], color="navy", linewidth=1.5, linestyle="--", label="Max gust")
axes[1].set_ylabel("Wind speed (kts)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# Relative humidity
axes[2].plot(t, sma_day["ure200h0"], color="mediumseagreen", linewidth=2)
axes[2].set_ylabel("Relative humidity (%)")
axes[2].set_ylim(0, 105)
axes[2].grid(True, alpha=0.3)

# Pressure
axes[3].plot(t, sma_day["prestah0"], color="darkorchid", linewidth=2)
axes[3].set_ylabel("Station pressure (hPa)")
axes[3].grid(True, alpha=0.3)

axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axes[3].xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.xticks(rotation=45)
plt.xlabel("Time (UTC)")

plt.tight_layout()

out_file_1 = os.path.join(out_dir, "meteo_5feb2026.png")
plt.savefig(out_file_1, dpi=150, bbox_inches="tight")
plt.show()

print(f"Figura salvata in: {out_file_1}")

# ─────────────────────────────────────────────
# FIGURE 2: THERMAL INVERSION PROFILE
# ─────────────────────────────────────────────

df_prof = pd.merge(
    sma_day[["reference_timestamp", "tre200h0"]],
    ueb_day[["reference_timestamp", "ta1towh0"]],
    on="reference_timestamp",
    how="inner"
)

print("Righe profilo:", len(df_prof))
print(df_prof.head())

# Quote reali / approssimate in metri sul livello del mare
z_sma = 550   # m asl circa SMA / Zürich-Fluntern
z_ueb = 870   # m asl circa UEB tower

fig, ax = plt.subplots(figsize=(6, 7))

# Seleziona 12 UTC
row_12 = df_prof[df_prof["reference_timestamp"].dt.hour == 12]

if row_12.empty:
    raise ValueError("Nessun dato trovato per le 12 UTC")

row_12 = row_12.iloc[0]

# Temperature osservate
t_sma = row_12["tre200h0"]
t_ueb = row_12["ta1towh0"]

# ─────────────────────────────────────────────
# PROFILO OSSERVATO
# ─────────────────────────────────────────────
ax.plot(
    [t_sma, t_ueb],
    [z_sma, z_ueb],
    linewidth=3,
    marker="o",
    markersize=8,
    color="red",
    label="Observed profile\n12:00 UTC"
)

# ─────────────────────────────────────────────
# DRY AND MOIST ADIABATIC LAPSE RATES
# ─────────────────────────────────────────────

# K/km = °C/km per differenze di temperatura
dry_lapse_rate = -9.81 / 1000   # °C/m
moist_lapse_rate = -6.5 / 1000  # °C/m

# Temperature teoriche alla quota UEB
t_dry_top = t_sma + dry_lapse_rate * (z_ueb - z_sma)
t_moist_top = t_sma + moist_lapse_rate * (z_ueb - z_sma)

# Dry adiabatic lapse rate
ax.plot(
    [t_sma, t_dry_top],
    [z_sma, z_ueb],
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Dry adiabatic lapse rate\n(-9.81 °C/km)"
)

# Moist adiabatic lapse rate
ax.plot(
    [t_sma, t_moist_top],
    [z_sma, z_ueb],
    color="black",
    linestyle=":",
    linewidth=1.5,
    label="Moist adiabatic lapse rate\n(-6.5 °C/km)"
)

# ─────────────────────────────────────────────
# FORMAT FIGURE
# ─────────────────────────────────────────────

ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Altitude (m asl)")

ax.set_title(
    "Temperature Profile – 5 February 2026, 12:00 UTC\nSMA 2 m vs UEB tower",
    fontweight="bold"
)

ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()

out_file_2 = os.path.join(out_dir, "inversion_profile_5feb2026_12UTC.png")
plt.savefig(out_file_2, dpi=150, bbox_inches="tight")
plt.show()

print(f"Figura salvata in: {out_file_2}")
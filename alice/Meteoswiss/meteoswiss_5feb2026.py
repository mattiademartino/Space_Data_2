import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df = pd.read_csv(
    "/Users/alice/Desktop/Space_Data_2/alice/Meteoswiss/ogd-smn_sma_h_recent.csv",
    sep=";",
    parse_dates=["reference_timestamp"],
    dayfirst=True,
    low_memory=False
)

df.columns = df.columns.str.strip()

day = df[df["reference_timestamp"].dt.date == pd.Timestamp("2026-02-05").date()].copy()

for col in ["tre200h0", "ure200h0", "prestah0", "fu3010h0", "fu3010h1"]:
    day[col] = pd.to_numeric(day[col], errors="coerce")

day["wind_kts"] = day["fu3010h0"] * 1.944
day["gust_kts"] = day["fu3010h1"] * 1.944

t = day["reference_timestamp"]

fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
fig.suptitle("Meteorological Conditions – Zürich/Fluntern (SMA)\n5 February 2026", fontsize=14, fontweight="bold")

axes[0].plot(t, day["tre200h0"], color="tomato", linewidth=2)
axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
axes[0].set_ylabel("Temperature (°C)")
axes[0].grid(True, alpha=0.3)
axes[0].fill_between(t, day["tre200h0"], 0, where=day["tre200h0"] < 0, alpha=0.12, color="blue")

axes[1].plot(t, day["wind_kts"], color="steelblue", linewidth=2, label="Mean wind")
axes[1].plot(t, day["gust_kts"], color="navy", linewidth=1.5, linestyle="--", label="Max gust")
axes[1].set_ylabel("Wind speed (kts)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, day["ure200h0"], color="mediumseagreen", linewidth=2)
axes[2].set_ylabel("Relative humidity (%)")
axes[2].set_ylim(0, 105)
axes[2].grid(True, alpha=0.3)

axes[3].plot(t, day["prestah0"], color="darkorchid", linewidth=2)
axes[3].set_ylabel("Station pressure (hPa)")
axes[3].grid(True, alpha=0.3)

axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
axes[3].xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.xticks(rotation=45)
plt.xlabel("Time (UTC)")

plt.tight_layout()
plt.savefig("./alice/Meteoswiss/img/meteo_5feb2026.png", dpi=150, bbox_inches="tight")
plt.show()
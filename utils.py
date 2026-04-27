"""
utils.py — Shared utilities for the GRASP/VAMOS CanSat data analysis.

Physics constants, ISA atmosphere helpers, data loaders, signal-processing
functions, flight-phase detection, and the Skew-T diagram routine all live here
so that main.ipynb remains a clean, readable narrative notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter
from scipy.signal import butter, filtfilt, welch

__version__ = "2.3.0"

# ── Repository layout ─────────────────────────────────────────────────────────
ROOT_DIR = Path(".")
DATA_DIR = ROOT_DIR / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external_dataset"
FIG_DIR = ROOT_DIR / "figures"

GRASP_CSV = DATA_DIR / "Probe_4_GRASP" / "run_0261 - with graphs.csv"
VAMOS_SCIENCE_CSV = DATA_DIR / "Probe_3_VAMOS" / "logs" / "science.csv"
VAMOS_WIND_CSV = DATA_DIR / "Probe_3_VAMOS" / "logs" / "wind.csv"
OBAMA_XLSX = DATA_DIR / "Probe_5_OBAMA" / "OBAMA_data_decoded.xlsx"
UWYO_CSV = EXTERNAL_DATA_DIR / "uwyo_06610_2026-02-05_12Z.csv"
SCAMSAT_DIR = DATA_DIR / "Probe_6_ScamSat" / "scamsat_data_upload" / "002"

FIGURE_GROUPS = {
    "plot_raw": FIG_DIR / "plot_raw",
    "data_quality": FIG_DIR / "data_quality",
    "flight_dynamics": FIG_DIR / "flight_dynamics",
    "cross_dataset": FIG_DIR / "cross_dataset",
    "signal_processing": FIG_DIR / "signal_processing",
    "external_dataset": FIG_DIR / "external_dataset",
    "scamsat": FIG_DIR / "scamsat",
    "exports": FIG_DIR / "exports",
}

# ── ISA standard-atmosphere constants (ICAO, troposphere layer) ───────────────
P0   = 101_325.0   # Pa      — sea-level reference pressure
T0_K = 288.15      # K       — sea-level reference temperature
L    = 6.5e-3      # K m⁻¹  — temperature lapse rate
G    = 9.80665     # m s⁻²  — standard gravity
RD   = 287.058     # J kg⁻¹ K⁻¹ — specific gas constant (dry air)

# ── WHO 2021 Air-Quality Guidelines ──────────────────────────────────────────
WHO_PM25_24H       = 15    # µg m⁻³ — PM₂.₅ 24-h mean limit
WHO_PM10_24H       = 45    # µg m⁻³ — PM₁₀  24-h mean limit
CO2_BACKGROUND_PPM = 420   # ppm    — global atmospheric background

# ── External sounding reference URLs ─────────────────────────────────────────
UWYO_SKEWT_URL = (
    "http://weather.uwyo.edu/wsgi/sounding"
    "?datetime=2026-02-05%2012:00:00&id=06610&type=PNG:SKEWT&src=BUFR"
)
_UWYO_CSV_URL = (
    "http://weather.uwyo.edu/wsgi/sounding"
    "?datetime=2026-02-05%2012:00:00&id=06610&type=TEXT:CSV&src=BUFR"
)

# ── Skew-T thermodynamic constants ───────────────────────────────────────────
_P_REF_HPA = 1000.0
_EPSILON    = 0.622
_RV         = 461.5
_CP_D       = 1004.0
_LV         = 2.5e6

_LEGACY_GRASP = Path("CanSat data utili") / "science_GRASP.csv"
_LEGACY_VAMOS_SCIENCE = Path("CanSat data utili") / "science_VAMOS.csv"
_LEGACY_VAMOS_WIND = Path("CanSat data utili") / "wind_VAMOS.csv"
_LEGACY_OBAMA = Path("CanSat data utili") / "OBAMA_data_decoded.xlsx"
_LEGACY_UWYO = Path("dati") / "uwyo_06610_2026-02-05_12Z.csv"


def ensure_repo_layout() -> None:
    """Create the canonical output folders used by the notebook and scripts."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for directory in FIGURE_GROUPS.values():
        directory.mkdir(parents=True, exist_ok=True)


def apply_plot_style() -> None:
    """Apply the shared matplotlib style used across the repository."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 10,
        "font.weight": "semibold",
        "axes.labelweight": "semibold",
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "#d7d7d7",
        "grid.linewidth": 0.8,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 0.9,
        "ytick.minor.width": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.8,
    })


def style_card_plot(
    ax: plt.Axes,
    *,
    show_legend: bool = False,
    add_texture: bool = True,
) -> None:
    """
    Style a compact saved plot so it stays legible inside report cards.

    The function removes the axes title, increases grid readability, adds a
    subtle paper-like texture, and optionally removes the legend to avoid
    clutter in small exported images.
    """
    ax.set_title("")
    ax.set_facecolor("white")
    ax.minorticks_on()
    ax.grid(True, which="major", color="#cfcfcf", linewidth=0.95, alpha=0.7)
    ax.grid(True, which="minor", color="#e7e7e7", linewidth=0.7, alpha=0.6)
    ax.tick_params(axis="both", labelsize=9, colors="#1f1f1f", width=1.2)

    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#222222")
        ax.spines[spine].set_linewidth(1.2)

    for axis in (ax.xaxis, ax.yaxis):
        if axis.get_scale() == "linear":
            try:
                axis.set_minor_locator(AutoMinorLocator())
            except ValueError:
                pass

    if add_texture:
        texture = Rectangle(
            (0, 0), 1, 1,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor=(0.0, 0.0, 0.0, 0.02),
            hatch="////",
            linewidth=0.0,
            zorder=0.1,
        )
        texture.set_in_layout(False)
        ax.add_patch(texture)

    legend = ax.get_legend()
    if legend is not None and not show_legend:
        legend.remove()


def _strip_export_text(fig: plt.Figure) -> None:
    """Remove labels and annotations from exported figures while keeping ticks."""
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")

    for ax in fig.axes:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        for text in list(ax.texts):
            text.set_text("")


def _resolve_existing_path(
    default_path: Path,
    *legacy_paths: Path,
    override: Path | None = None,
) -> Path:
    """Resolve the first available path among canonical, legacy, or override."""
    candidates: list[Path] = []
    if override is not None:
        override = Path(override)
        if override.is_dir():
            candidates.append(override / default_path.name)
            candidates.extend(override / legacy.name for legacy in legacy_paths)
        else:
            candidates.append(override)
    candidates.append(default_path)
    candidates.extend(legacy_paths)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    locations = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not find the requested dataset in any of:\n{locations}")


def _first_existing_path(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _load_scamsat_signal(
    filename: str,
    *,
    dtype: str,
    scale: float,
    time_span_s: float,
    sync_shift_s: float = 0.0,
    run_dir: Path | None = None,
) -> pd.DataFrame:
    """Load a single ScamSat binary channel and attach its reconstructed time axis."""
    base_dir = Path(run_dir) if run_dir else SCAMSAT_DIR
    arr = np.fromfile(base_dir / filename, dtype=dtype).astype(float) * scale
    t_s = np.linspace(1.0, time_span_s, len(arr)) + sync_shift_s
    return pd.DataFrame({"t_s": t_s, "value": arr})


ensure_repo_layout()


# =============================================================================
# ISA Standard Atmosphere
# =============================================================================

def isa_alt(p_pa: np.ndarray) -> np.ndarray:
    """Barometric altitude (m ASL) from absolute pressure (Pa), ISA troposphere."""
    return (T0_K / L) * (1.0 - (np.asarray(p_pa, dtype=float) / P0) ** (RD * L / G))


def isa_press(h_m: np.ndarray) -> np.ndarray:
    """ISA pressure (Pa) at altitude h (m ASL)."""
    return P0 * (1.0 - L * np.asarray(h_m, dtype=float) / T0_K) ** (G / (RD * L))


def isa_temp_c(h_m: np.ndarray) -> np.ndarray:
    """ISA temperature (°C) at altitude h (m ASL)."""
    return T0_K - L * np.asarray(h_m, dtype=float) - 273.15


def baro_altitude_agl(p_hpa: np.ndarray, p0_hpa: float) -> np.ndarray:
    """
    Barometric altitude above ground level (m) using a local surface reference.

    Uses the simplified hypsometric formula:  h = 44 330 · [1 − (P/P₀)^(1/5.255)]
    This is appropriate for the 0–3 km range spanned by a typical CanSat flight.
    """
    return 44_330.0 * (1.0 - (np.asarray(p_hpa, dtype=float) / p0_hpa) ** (1.0 / 5.255))


# =============================================================================
# Signal Processing
# =============================================================================

def resample_uniform(
    t: np.ndarray,
    y: np.ndarray,
    fs_new: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample an irregularly sampled time series onto a uniform grid.

    Steps:
      1. Drop NaN / ±Inf samples.
      2. Sort by time and deduplicate.
      3. Interpolate linearly onto a regular grid at *fs_new* Hz.

    Parameters
    ----------
    t      : time axis (seconds), possibly non-uniform
    y      : signal values
    fs_new : target sample rate (Hz)

    Returns
    -------
    (t_uniform, y_uniform)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(t) & np.isfinite(y)
    t, y = t[keep], y[keep]
    order = np.argsort(t)
    t, y = t[order], y[order]
    t_uniq, idx = np.unique(t, return_index=True)
    y_uniq = y[idx]
    t_out = np.arange(t_uniq[0], t_uniq[-1], 1.0 / fs_new)
    y_out = np.interp(t_out, t_uniq, y_uniq)
    return t_out, y_out


def butter_lowpass(data: np.ndarray, cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter (forward-backward, no phase shift).

    Parameters
    ----------
    data      : input signal (must already be on a uniform grid)
    cutoff_hz : −3 dB cutoff frequency (Hz)
    fs        : sample rate of *data* (Hz)
    order     : filter order (default 4 → −80 dB/decade roll-off)
    """
    b, a = butter(order, cutoff_hz / (0.5 * fs), btype="low")
    return filtfilt(b, a, data)


def detrend_linear(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Remove a least-squares linear trend from signal x(t)."""
    return x - np.polyval(np.polyfit(t, x, 1), t)


def compute_welch_psd(
    t: np.ndarray,
    y: np.ndarray,
    fs: float,
    nperseg: int,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Estimate PSD with Welch's method after uniform resampling.

    The input (t, y) may be irregularly sampled — the function resamples to
    *fs* Hz internally before calling `scipy.signal.welch`.

    Returns
    -------
    freqs      : frequency axis (Hz)
    psd        : power spectral density
    n_samples  : number of samples after resampling
    dof        : approximate degrees of freedom (2 · n_segments · 2.2)
    """
    if noverlap is None:
        noverlap = nperseg // 2
    t_u, y_u = resample_uniform(t, y, fs)
    y_u -= y_u.mean()
    nperseg_use  = min(nperseg, len(y_u))
    noverlap_use = min(noverlap, max(nperseg_use - 1, 0))
    freqs, psd = welch(
        y_u, fs=fs, window="hann",
        nperseg=nperseg_use, noverlap=noverlap_use,
        detrend="linear", scaling="density",
    )
    dof = 2.0 * (len(y_u) / max(nperseg_use, 1)) * 2.2
    return freqs, psd, len(y_u), dof


# =============================================================================
# Wind Utilities
# =============================================================================

def met_to_uv(
    direction_deg: np.ndarray,
    speed_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert meteorological wind (direction-from, speed) to Cartesian (u, v).

    Convention: *direction* is the compass bearing the wind blows *from*,
    so a northerly wind (from 360°) gives u = 0, v < 0.

    u > 0 → eastward;  v > 0 → northward.
    """
    ang = np.deg2rad(np.asarray(direction_deg, dtype=float))
    spd = np.asarray(speed_ms, dtype=float)
    u   = -spd * np.sin(ang)
    v   = -spd * np.cos(ang)
    return u, v


# =============================================================================
# Flight-phase Detection
# =============================================================================

def detect_vamos_drop(vamos_df: pd.DataFrame) -> dict:
    """
    Identify the descent phase (apogee → landing) from the VAMOS pressure record.

    Algorithm
    ---------
    1. Estimate ground-level pressure p₀ as the median of the first 100 and
       last 500 samples (robust to in-flight extremes).
    2. Smooth pressure with a rolling-5 median, then compute dp/dt.
    3. Find contiguous segments where dp/dt > 0.1 hPa s⁻¹ (pressure rising →
       CanSat descending).  The segment with the largest pressure increase is
       the main drop.
    4. Walk back from the drop start to find the true pressure minimum (apogee).
    5. Walk forward from the drop end to find where pressure returns to p₀
       (landing).

    Parameters
    ----------
    vamos_df : DataFrame with columns 't_s' and 'press_hPa'

    Returns
    -------
    Dictionary with keys:
      p0_hPa           : estimated ground-level pressure (hPa)
      h_agl            : barometric altitude AGL array (m), same length as vamos_df
      apogee_idx        : row index of peak altitude
      landing_idx       : row index of landing
      t_apogee_s        : absolute time of apogee (s)
      t_landing_s       : absolute time of landing (s)
      h_peak_m          : peak altitude AGL (m)
      drop_duration_s   : seconds from apogee to landing
      descent_rate_mps  : mean descent rate (m s⁻¹, positive = downward)
      drop_mask         : boolean array, True during the drop phase
    """
    p = vamos_df["press_hPa"].to_numpy(dtype=float)
    t = vamos_df["t_s"].to_numpy(dtype=float)

    p0    = (np.median(p[:100]) + np.median(p[-500:])) / 2
    h_agl = baro_altitude_agl(p, p0)

    p_smooth = pd.Series(p).rolling(5, center=True, min_periods=1).median().to_numpy()
    dpdt     = np.gradient(p_smooth) / np.gradient(t)

    rising = np.where(dpdt > 0.1)[0]
    if len(rising) == 0:
        apogee_idx  = int(np.argmin(p))
        landing_idx = len(p) - 1
    else:
        gaps   = np.where(np.diff(rising) > 15)[0]
        starts = np.concatenate([[rising[0]], rising[gaps + 1]])
        ends   = np.concatenate([rising[gaps], [rising[-1]]])
        dps    = np.array([p[e] - p[s] for s, e in zip(starts, ends)])
        best   = int(np.argmax(dps))
        ds, de = starts[best], ends[best]

        look_back  = max(0, ds - 60)
        apogee_idx = look_back + int(np.argmin(p[look_back : ds + 1]))

        look_fwd    = min(len(p), de + 120)
        post_ground = np.where(p[de:look_fwd] > p0 - 0.5)[0]
        landing_idx = de + (post_ground[0] if len(post_ground) else 0)

    drop_mask = (t >= t[apogee_idx]) & (t <= t[landing_idx])
    dt_drop   = max(t[landing_idx] - t[apogee_idx], 1.0)

    return {
        "p0_hPa":           float(p0),
        "h_agl":            h_agl,
        "apogee_idx":       int(apogee_idx),
        "landing_idx":      int(landing_idx),
        "t_apogee_s":       float(t[apogee_idx]),
        "t_landing_s":      float(t[landing_idx]),
        "h_peak_m":         float(h_agl[apogee_idx]),
        "drop_duration_s":  float(t[landing_idx] - t[apogee_idx]),
        "descent_rate_mps": float((h_agl[apogee_idx] - h_agl[landing_idx]) / dt_drop),
        "drop_mask":        drop_mask,
    }


# =============================================================================
# Data Loaders
# =============================================================================

def load_grasp(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load the GRASP atmospheric payload data.

    Columns returned: t_ms, temp_C, press_Pa, alt_m, pm25, pm10, t_s, t_rel.
    Note: row 0 is a sensor-init artefact and is NOT removed here — the caller
    is responsible for filtering it (typically `df.iloc[1:]`).
    """
    src = _resolve_existing_path(GRASP_CSV, _LEGACY_GRASP, override=data_dir)
    df = pd.read_csv(
        src,
        usecols=[0, 2, 3, 4, 5, 6],
        names=["t_ms", "temp_C", "press_Pa", "alt_m", "pm25", "pm10"],
        header=0, skipfooter=8, engine="python",
    )
    df = df.dropna().astype(float)
    df["t_s"]   = df["t_ms"] / 1000.0
    df["t_rel"] = df["t_s"] - df["t_s"].iat[0]
    return df.reset_index(drop=True)


def load_vamos_science(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load the VAMOS science payload data.

    Cleans repeated-header rows (firmware bug), converts timestamps, and adds
    a barometric altitude column (alt_baro, m ASL) via the ISA formula.

    Columns returned: t_ms, co2_ppm, temp_C, press_hPa, t_s, t_rel, alt_baro.
    """
    src = _resolve_existing_path(VAMOS_SCIENCE_CSV, _LEGACY_VAMOS_SCIENCE, override=data_dir)
    df = pd.read_csv(src)
    df.columns = ["t_ms", "co2_ppm", "temp_C", "press_hPa"]
    df = (
        df[df["t_ms"] != "timestamp_ms"]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .reset_index(drop=True)
    )
    df["t_s"]      = df["t_ms"] / 1000.0
    df["t_rel"]    = df["t_s"] - df["t_s"].iat[0]
    df["alt_baro"] = isa_alt(df["press_hPa"].values * 100.0)
    return df


def load_vamos_wind(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load the VAMOS wind payload data.

    Handles a mid-file timestamp reset (microcontroller reboot) by discarding
    the segment before the last reset and keeping only the continuous tail.
    Adds vector wind speed and vector acceleration columns.

    Columns returned: t_ms, wind_acc, wind_dir, wind_spd, x_acc, y_acc,
                      x_mps, y_mps, tumbling, t_s, t_rel,
                      wind_acc_vec, wind_spd_vec.
    """
    src = _resolve_existing_path(VAMOS_WIND_CSV, _LEGACY_VAMOS_WIND, override=data_dir)
    df = pd.read_csv(src)
    df.columns = ["t_ms", "wind_acc", "wind_dir", "wind_spd",
                  "x_acc", "y_acc", "x_mps", "y_mps", "tumbling"]
    df = (
        df[df["t_ms"] != "timestamp_ms"]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .reset_index(drop=True)
    )
    df["t_s"] = df["t_ms"] / 1000.0

    resets = np.where(np.diff(df["t_s"].values) < 0)[0]
    if len(resets):
        df = df.iloc[resets[-1] + 1 :].copy()

    df = df.reset_index(drop=True)
    df["t_s"]          = df["t_ms"] / 1000.0
    df["t_rel"]        = df["t_s"] - df["t_s"].iat[0]
    df["wind_acc_vec"] = np.hypot(df["x_acc"], df["y_acc"])
    df["wind_spd_vec"] = np.hypot(df["x_mps"], df["y_mps"])
    return df


def load_obama(data_dir: Path | None = None) -> pd.DataFrame:
    """Load OBAMA_data_decoded.xlsx (external CanSat reference dataset)."""
    src = _resolve_existing_path(OBAMA_XLSX, _LEGACY_OBAMA, override=data_dir)
    df = pd.read_excel(src, sheet_name="decoded")
    return df.apply(pd.to_numeric, errors="coerce")


def load_uwyo_sounding(
    csv_path: Path | None = None,
    url: str | None = None,
) -> pd.DataFrame:
    """
    Load the University of Wyoming upper-air sounding (station 06610, Payerne).

    Falls back to the remote URL if the local CSV does not exist.
    """
    local_path = _first_existing_path(
        *( [Path(csv_path)] if csv_path is not None else [] ),
        UWYO_CSV,
        _LEGACY_UWYO,
    )
    src: Path | str = local_path if local_path is not None else (url or _UWYO_CSV_URL)
    if isinstance(src, Path) and not src.exists():
        src = url or _UWYO_CSV_URL
    df = pd.read_csv(src)
    return df.rename(columns={
        "geopotential height_m":   "height_m",
        "dew point temperature_C": "dewpoint_C",
        "wind direction_degree":   "wind_dir_deg",
        "wind speed_m/s":          "wind_spd_ms",
    })


def load_era5_profile(
    nc_path: Path | None = None,
    *,
    station_elevation_m: float = 448.0,
) -> pd.DataFrame | None:
    """
    Load an ERA5 pressure-level profile stored in data/external_dataset.

    Returns a tidy DataFrame with altitude AGL, pressure, temperature, RH and
    wind components. Returns None when no local NetCDF file is available.
    """
    import xarray as xr

    if nc_path is not None:
        src = Path(nc_path)
    else:
        src = _first_existing_path(
            EXTERNAL_DATA_DIR / "era5_dubendorf_20260206.nc",
            *sorted(EXTERNAL_DATA_DIR.glob("era5*.nc")),
        )
    if src is None:
        return None

    ds = xr.open_dataset(src)
    if {"latitude", "longitude"}.issubset(ds.dims):
        ds = ds.mean(dim=["latitude", "longitude"])

    time_coord = "time" if "time" in ds.coords else "valid_time" if "valid_time" in ds.coords else None
    if time_coord is not None:
        ds = ds.isel({time_coord: 0})
        selected_time: Any = ds[time_coord].values
    else:
        selected_time = None

    p_hpa = np.asarray(ds["level"].values, dtype=float)
    t_c = np.asarray(ds["t"].values, dtype=float) - 273.15
    z_asl = np.asarray(ds["z"].values, dtype=float) / G
    rh = np.asarray(ds["r"].values, dtype=float) if "r" in ds else np.full_like(t_c, np.nan)
    u = np.asarray(ds["u"].values, dtype=float) if "u" in ds else np.full_like(t_c, np.nan)
    v = np.asarray(ds["v"].values, dtype=float) if "v" in ds else np.full_like(t_c, np.nan)
    wspd = np.hypot(u, v)
    h_agl = z_asl - station_elevation_m

    df = pd.DataFrame({
        "pressure_hPa": p_hpa,
        "temperature_C": t_c,
        "height_asl_m": z_asl,
        "height_agl_m": h_agl,
        "relative_humidity_pct": rh,
        "u_ms": u,
        "v_ms": v,
        "wind_spd_ms": wspd,
    }).sort_values("height_agl_m").reset_index(drop=True)
    df.attrs["source_path"] = str(src)
    if selected_time is not None:
        df.attrs["selected_time"] = str(selected_time)
    return df


def load_scamsat_bundle(run_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Load the ScamSat binary products as a dict of tidy time series.

    Returned keys: altitude, pressure, temperature, pm25, pm10, acceleration, fft.
    """
    base_dir = Path(run_dir) if run_dir else SCAMSAT_DIR
    bundle = {
        "altitude": _load_scamsat_signal(
            "alt.txt", dtype="uint16", scale=0.1, time_span_s=3550.0, run_dir=base_dir
        ).rename(columns={"value": "altitude_m"}),
        "pressure": _load_scamsat_signal(
            "press.txt", dtype="float32", scale=1.0, time_span_s=3550.0, run_dir=base_dir
        ).rename(columns={"value": "press_Pa"}),
        "temperature": _load_scamsat_signal(
            "temp.txt", dtype="float32", scale=1.0, time_span_s=3550.0, run_dir=base_dir
        ).rename(columns={"value": "temp_C"}),
        "pm25": _load_scamsat_signal(
            "pm2_5.txt", dtype="uint16", scale=0.1, time_span_s=3550.0, run_dir=base_dir
        ).rename(columns={"value": "pm25"}),
        "pm10": _load_scamsat_signal(
            "pm10.txt", dtype="uint16", scale=0.1, time_span_s=3550.0, run_dir=base_dir
        ).rename(columns={"value": "pm10"}),
        "acceleration": _load_scamsat_signal(
            "accel.txt", dtype="int16", scale=1 / 1000, time_span_s=3450.0, sync_shift_s=43.0, run_dir=base_dir
        ).rename(columns={"value": "accel_g"}),
        "fft": _load_scamsat_signal(
            "fft.txt", dtype="float32", scale=1.0, time_span_s=3650.0, sync_shift_s=-36.0, run_dir=base_dir
        ).rename(columns={"value": "fft_amplitude"}),
    }

    for frame in bundle.values():
        frame["t_rel"] = frame["t_s"] - frame["t_s"].iat[0]

    bundle["altitude"]["altitude_agl_m"] = bundle["altitude"]["altitude_m"] - 448.0
    bundle["pressure"]["alt_baro_m"] = isa_alt(bundle["pressure"]["press_Pa"].values)
    bundle["temperature"]["altitude_agl_m"] = np.interp(
        bundle["temperature"]["t_s"],
        bundle["altitude"]["t_s"],
        bundle["altitude"]["altitude_agl_m"],
    )
    bundle["pm25"]["altitude_agl_m"] = np.interp(
        bundle["pm25"]["t_s"],
        bundle["altitude"]["t_s"],
        bundle["altitude"]["altitude_agl_m"],
    )
    bundle["pm10"]["altitude_agl_m"] = np.interp(
        bundle["pm10"]["t_s"],
        bundle["altitude"]["t_s"],
        bundle["altitude"]["altitude_agl_m"],
    )
    return bundle


# =============================================================================
# Dataset Summary
# =============================================================================

def print_dataset_summary(
    grasp,
    vamos,
    wind,
    obama,
    uwyo,
    vamos_drop,
    vamos_conc,
    era5: pd.DataFrame | None = None,
    scamsat: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Print a concise summary table of all loaded datasets."""
    fs_g = 1000.0 / np.median(np.diff(grasp["t_ms"].values))
    fs_v = 1000.0 / np.median(np.diff(vamos["t_ms"].values))
    fs_w = 1000.0 / np.median(np.diff(wind["t_ms"].values))

    print("─── Dataset summary ────────────────────────────────────────────────────────")
    print(f"  GRASP          : {len(grasp):6d} rows | {grasp['t_rel'].iat[-1]:.1f} s | fs ≈ {fs_g:.2f} Hz")
    print(f"  VAMOS-science  : {len(vamos):6d} rows | {vamos['t_rel'].iat[-1]/60:.1f} min | fs ≈ {fs_v:.2f} Hz")
    print(f"  VAMOS-wind     : {len(wind):6d} rows | {wind['t_rel'].iat[-1]/60:.1f} min | fs ≈ {fs_w:.2f} Hz")
    print(f"  OBAMA (ext.)   : {obama['Time_s'].count():6.0f} rows | time up to {obama['Time_s'].max():.0f} s")
    print(f"  UWYO 06610     : {len(uwyo):6d} levels | sounding 2026-02-05 12:00 UTC (Payerne, CH)")
    if era5 is not None:
        label = era5.attrs.get("selected_time", "local NetCDF")
        print(f"  ERA5 (ext.)    : {len(era5):6d} levels | {label}")
    if scamsat is not None:
        print(f"  ScamSat        : {len(scamsat['altitude']):6d} altitude rows | "
              f"{len(scamsat['temperature']):6d} temperature rows")
    print(f"  VAMOS ∩ GRASP  : {len(vamos_conc):6d} rows concurrent with the GRASP flight")
    print(f"  Drop phase     : apogee at {vamos_drop['t_apogee_s']:.0f} s → landing at"
          f" {vamos_drop['t_landing_s']:.0f} s | "
          f"peak ≈ {vamos_drop['h_peak_m']:.0f} m AGL | "
          f"duration {vamos_drop['drop_duration_s']:.0f} s")
    print("────────────────────────────────────────────────────────────────────────────")


# =============================================================================
# Figure helpers
# =============================================================================

def save_figure(fig: plt.Figure, filename: str, *, strip_text: bool = True, **kwargs) -> Path:
    """Save *fig* under figures/<filename> and return the written path."""
    ensure_repo_layout()
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
    if strip_text:
        _strip_export_text(fig)
    path = FIG_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults = {"bbox_inches": "tight", "facecolor": "white", "edgecolor": "white"}
    defaults.update(kwargs)
    fig.savefig(path, **defaults)
    return path


# =============================================================================
# Skew-T / Log-P diagram (UWYO-style, no external dependencies)
# =============================================================================

def _skew_x(temperature_c, pressure_hpa, skew: float) -> np.ndarray:
    return np.asarray(temperature_c) + skew * np.log(_P_REF_HPA / np.asarray(pressure_hpa))


def _sat_vapor_pressure_hpa(temperature_c) -> np.ndarray:
    tc = np.asarray(temperature_c, dtype=float)
    return 6.112 * np.exp(17.67 * tc / (tc + 243.5))


def _mixing_ratio_from_dewpoint(pressure_hpa, dewpoint_c) -> np.ndarray:
    e = _sat_vapor_pressure_hpa(dewpoint_c)
    return _EPSILON * e / np.maximum(np.asarray(pressure_hpa, dtype=float) - e, 1e-6)


def _dewpoint_from_mixing_ratio(pressure_hpa, mixing_ratio_gkg: float) -> np.ndarray:
    ratio   = mixing_ratio_gkg / 1000.0
    vp      = ratio * np.asarray(pressure_hpa, dtype=float) / (_EPSILON + ratio)
    log_t   = np.log(np.maximum(vp, 1e-6) / 6.112)
    return 243.5 * log_t / (17.67 - log_t)


def _dry_adiabat_c(theta_k: float, pressure_hpa) -> np.ndarray:
    return (theta_k * (np.asarray(pressure_hpa, dtype=float) / _P_REF_HPA)
            ** (RD / _CP_D) - 273.15)


def _moist_lapse_rate(T_k: float, pressure_hpa: float) -> float:
    w_sat = _mixing_ratio_from_dewpoint(np.array([pressure_hpa]),
                                        np.array([T_k - 273.15]))[0]
    num   = G * (1.0 + _LV * w_sat / (RD * T_k))
    den   = _CP_D + (_LV**2 * w_sat * _EPSILON) / (_RV * T_k**2)
    return num / den


def _moist_adiabat_c(start_temp_c: float, pressure_hpa) -> np.ndarray:
    pressure_hpa = np.asarray(pressure_hpa, dtype=float)
    Tk           = np.empty_like(pressure_hpa)
    Tk[0]        = start_temp_c + 273.15
    for i in range(1, len(pressure_hpa)):
        p_prev = pressure_hpa[i - 1]
        dp     = (pressure_hpa[i] - p_prev) * 100.0          # hPa → Pa
        w_sat  = _mixing_ratio_from_dewpoint(np.array([p_prev]),
                                             np.array([Tk[i - 1] - 273.15]))[0]
        gamma  = _moist_lapse_rate(Tk[i - 1], p_prev)
        vT     = Tk[i - 1] * (1.0 + 0.61 * w_sat)
        dTdp   = gamma * RD * vT / (p_prev * 100.0 * G)
        Tk[i]  = Tk[i - 1] + dTdp * dp
    return Tk - 273.15


def _configure_pressure_axis(ax: plt.Axes) -> None:
    major = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
    ax.set_yscale("log")
    ax.set_ylim(1050, 100)
    ax.yaxis.set_major_locator(FixedLocator(major))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):d}"))
    ax.grid(False)


def plot_uwyo_skewt(
    ax: plt.Axes,
    sounding_df: pd.DataFrame,
    *,
    skew: float = 35.0,
    xlim: tuple[float, float] = (-40.0, 60.0),
    barb_stride: int = 75,
) -> None:
    """
    Draw a Skew-T / log-P panel styled after the UWYO online diagram.

    Background lines include: isotherms, isobars, dry adiabats (red dashed),
    moist adiabats (blue dashed), and mixing-ratio lines (green dashed).
    Wind barbs are plotted at every *barb_stride*-th level.

    Parameters
    ----------
    ax           : Axes to draw on (will be reconfigured to log-P scale)
    sounding_df  : DataFrame with columns pressure_hPa, temperature_C,
                   dewpoint_C, u_ms, v_ms (already computed from met convention)
    skew         : skewness angle (°C per log-decade of pressure)
    xlim         : x-axis temperature range (°C)
    barb_stride  : plot one barb every this many levels
    """
    prof = sounding_df.sort_values("pressure_hPa", ascending=False).copy()
    p    = prof["pressure_hPa"].to_numpy(dtype=float)
    T    = prof["temperature_C"].to_numpy(dtype=float)
    Td   = prof["dewpoint_C"].to_numpy(dtype=float)
    u    = prof["u_ms"].to_numpy(dtype=float)
    v    = prof["v_ms"].to_numpy(dtype=float)

    p_grid = np.geomspace(1000.0, 100.0, 180)
    _configure_pressure_axis(ax)
    ax.set_xlim(*xlim)
    ax.set_facecolor("white")

    # ── Background grid ───────────────────────────────────────────────────────
    for p_line in [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]:
        ax.axhline(p_line, color="0.75", lw=0.8, zorder=0)
    for T0 in np.arange(-40, 61, 10):
        ax.plot(_skew_x(np.full_like(p_grid, T0), p_grid, skew), p_grid,
                color="0.75", lw=0.8, zorder=0)
    for theta_k in np.arange(260, 450, 10):
        ax.plot(_skew_x(_dry_adiabat_c(theta_k, p_grid), p_grid, skew), p_grid,
                color="#ff6b6b", lw=1.0, ls="--", alpha=0.9, zorder=0)
    for Ts in np.arange(-30, 46, 10):
        ax.plot(_skew_x(_moist_adiabat_c(Ts, p_grid), p_grid, skew), p_grid,
                color="#6a7bff", lw=1.0, ls="--", alpha=0.9, zorder=0)
    for ratio in [0.4, 1, 2, 4, 7, 10, 16, 24, 32]:
        p_mix  = np.geomspace(1000.0, 500.0, 80)
        td_mix = _dewpoint_from_mixing_ratio(p_mix, ratio)
        ax.plot(_skew_x(td_mix, p_mix, skew), p_mix,
                color="#4caf50", lw=1.0, ls="--", alpha=0.95, zorder=0)
        ax.text(_skew_x(td_mix[0], p_mix[0], skew), 985,
                f"{ratio:.1f}", color="#43a047", fontsize=7, ha="center", va="bottom")

    # ── Temperature and dew point profiles ───────────────────────────────────
    ax.plot(_skew_x(T,  p, skew), p, color="black", lw=1.8, zorder=4, label="Temperature")
    ax.plot(_skew_x(Td, p, skew), p, color="black", lw=1.8, ls="--", zorder=4, label="Dew point")

    # ── Altitude labels (m) ───────────────────────────────────────────────────
    for p_lbl, alt in [(1000, "92"), (925, "586"), (850, "1268"), (700, "2817"),
                       (500, "5380"), (400, "6969"), (300, "8912"), (250, "10097"),
                       (200, "11557"), (150, "13447"), (100, "16093")]:
        if p.min() <= p_lbl <= p.max():
            ax.text(xlim[0] + 3.0, p_lbl, alt, color="0.45", fontsize=7, va="center")

    # ── Wind barbs ────────────────────────────────────────────────────────────
    bp  = p[::barb_stride]
    bx  = np.full_like(bp, xlim[1] - 3.0)
    kts = 1.94384
    ax.barbs(bx, bp, u[::barb_stride] * kts, v[::barb_stride] * kts,
             length=5.0, linewidth=0.5, barbcolor="black", flagcolor="black", zorder=5)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Pressure (hPa)")

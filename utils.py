"""
utils.py — Shared utilities for the GRASP/VAMOS CanSat data analysis.

Physics constants, ISA atmosphere helpers, data loaders, signal-processing
functions, flight-phase detection, and the Skew-T diagram routine all live here
so that main.ipynb remains a clean, readable narrative notebook.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
from scipy.signal import butter, filtfilt, welch

__version__ = "2.1.0"

# ── Output directory ──────────────────────────────────────────────────────────
FIG_DIR = Path("figures")

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

# ── Default local data paths ──────────────────────────────────────────────────
_DATA     = Path("CanSat data utili")
_UWYO_CSV = Path("dati/uwyo_06610_2026-02-05_12Z.csv")

# ── Skew-T thermodynamic constants ───────────────────────────────────────────
_P_REF_HPA = 1000.0
_EPSILON    = 0.622
_RV         = 461.5
_CP_D       = 1004.0
_LV         = 2.5e6


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
    Load science_GRASP.csv.

    Columns returned: t_ms, temp_C, press_Pa, alt_m, pm25, pm10, t_s, t_rel.
    Note: row 0 is a sensor-init artefact and is NOT removed here — the caller
    is responsible for filtering it (typically `df.iloc[1:]`).
    """
    root = Path(data_dir) if data_dir else _DATA
    df = pd.read_csv(
        root / "science_GRASP.csv",
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
    Load science_VAMOS.csv.

    Cleans repeated-header rows (firmware bug), converts timestamps, and adds
    a barometric altitude column (alt_baro, m ASL) via the ISA formula.

    Columns returned: t_ms, co2_ppm, temp_C, press_hPa, t_s, t_rel, alt_baro.
    """
    root = Path(data_dir) if data_dir else _DATA
    df = pd.read_csv(root / "science_VAMOS.csv")
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
    Load wind_VAMOS.csv.

    Handles a mid-file timestamp reset (microcontroller reboot) by discarding
    the segment before the last reset and keeping only the continuous tail.
    Adds vector wind speed and vector acceleration columns.

    Columns returned: t_ms, wind_acc, wind_dir, wind_spd, x_acc, y_acc,
                      x_mps, y_mps, tumbling, t_s, t_rel,
                      wind_acc_vec, wind_spd_vec.
    """
    root = Path(data_dir) if data_dir else _DATA
    df = pd.read_csv(root / "wind_VAMOS.csv")
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
    root = Path(data_dir) if data_dir else _DATA
    df = pd.read_excel(root / "OBAMA_data_decoded.xlsx", sheet_name="decoded")
    return df.apply(pd.to_numeric, errors="coerce")


def load_uwyo_sounding(
    csv_path: Path | None = None,
    url: str | None = None,
) -> pd.DataFrame:
    """
    Load the University of Wyoming upper-air sounding (station 06610, Payerne).

    Falls back to the remote URL if the local CSV does not exist.
    """
    src: Path | str = csv_path or _UWYO_CSV
    if not Path(src).exists():
        src = url or _UWYO_CSV_URL
    df = pd.read_csv(src)
    return df.rename(columns={
        "geopotential height_m":   "height_m",
        "dew point temperature_C": "dewpoint_C",
        "wind direction_degree":   "wind_dir_deg",
        "wind speed_m/s":          "wind_spd_ms",
    })


# =============================================================================
# Dataset Summary
# =============================================================================

def print_dataset_summary(
    grasp, vamos, wind, obama, uwyo, vamos_drop, vamos_conc
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
    print(f"  VAMOS ∩ GRASP  : {len(vamos_conc):6d} rows concurrent with the GRASP flight")
    print(f"  Drop phase     : apogee at {vamos_drop['t_apogee_s']:.0f} s → landing at"
          f" {vamos_drop['t_landing_s']:.0f} s | "
          f"peak ≈ {vamos_drop['h_peak_m']:.0f} m AGL | "
          f"duration {vamos_drop['drop_duration_s']:.0f} s")
    print("────────────────────────────────────────────────────────────────────────────")


# =============================================================================
# Figure helpers
# =============================================================================

def save_figure(fig: plt.Figure, filename: str, **kwargs) -> Path:
    """Save *fig* to Report/Img/<filename> and return the written path."""
    path = FIG_DIR / filename
    defaults = {"bbox_inches": "tight"}
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

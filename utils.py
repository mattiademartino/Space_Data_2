from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

FIG_DIR = Path("Report/Img")
FIG_DIR.mkdir(parents=True, exist_ok=True)

P_REF_HPA = 1000.0
EPSILON = 0.622
RD = 287.05
RV = 461.5
CP_D = 1004.0
G = 9.80665
LV = 2.5e6


def save_figure(fig: plt.Figure, filename: str, **kwargs) -> Path:
    """Save a figure in Report/Img and return the written path."""
    path = FIG_DIR / filename
    defaults = {"bbox_inches": "tight"}
    defaults.update(kwargs)
    fig.savefig(path, **defaults)
    return path


def _skew_x(temperature_c: np.ndarray, pressure_hpa: np.ndarray, skew: float) -> np.ndarray:
    return np.asarray(temperature_c) + skew * np.log(P_REF_HPA / np.asarray(pressure_hpa))


def _sat_vapor_pressure_hpa(temperature_c: np.ndarray) -> np.ndarray:
    temperature_c = np.asarray(temperature_c, dtype=float)
    return 6.112 * np.exp(17.67 * temperature_c / (temperature_c + 243.5))


def _mixing_ratio_from_dewpoint(pressure_hpa: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    e = _sat_vapor_pressure_hpa(dewpoint_c)
    return EPSILON * e / np.maximum(np.asarray(pressure_hpa, dtype=float) - e, 1e-6)


def _dewpoint_from_mixing_ratio(pressure_hpa: np.ndarray, mixing_ratio_gkg: float) -> np.ndarray:
    ratio = mixing_ratio_gkg / 1000.0
    vapor_pressure = ratio * np.asarray(pressure_hpa, dtype=float) / (EPSILON + ratio)
    log_term = np.log(np.maximum(vapor_pressure, 1e-6) / 6.112)
    return 243.5 * log_term / (17.67 - log_term)


def _dry_adiabat_c(theta_k: float, pressure_hpa: np.ndarray) -> np.ndarray:
    return theta_k * (np.asarray(pressure_hpa, dtype=float) / P_REF_HPA) ** (RD / CP_D) - 273.15


def _moist_lapse_rate_km(T_k: float, pressure_hpa: float) -> float:
    temperature_c = T_k - 273.15
    w_sat = _mixing_ratio_from_dewpoint(np.array([pressure_hpa]), np.array([temperature_c]))[0]
    numerator = G * (1.0 + LV * w_sat / (RD * T_k))
    denominator = CP_D + (LV**2 * w_sat * EPSILON) / (RV * T_k**2)
    return numerator / denominator


def _moist_adiabat_c(start_temp_c: float, pressure_hpa: np.ndarray) -> np.ndarray:
    pressure_hpa = np.asarray(pressure_hpa, dtype=float)
    temperature_k = np.empty_like(pressure_hpa)
    temperature_k[0] = start_temp_c + 273.15

    for idx in range(1, len(pressure_hpa)):
        p_prev_hpa = pressure_hpa[idx - 1]
        p_prev_pa = p_prev_hpa * 100.0
        p_curr_pa = pressure_hpa[idx] * 100.0
        dp = p_curr_pa - p_prev_pa
        w_sat = _mixing_ratio_from_dewpoint(np.array([p_prev_hpa]), np.array([temperature_k[idx - 1] - 273.15]))[0]
        gamma_m = _moist_lapse_rate_km(temperature_k[idx - 1], p_prev_hpa)
        virtual_temperature = temperature_k[idx - 1] * (1.0 + 0.61 * w_sat)
        dTdp = gamma_m * RD * virtual_temperature / (p_prev_pa * G)
        temperature_k[idx] = temperature_k[idx - 1] + dTdp * dp

    return temperature_k - 273.15


def _configure_pressure_axis(ax: plt.Axes) -> None:
    major_ticks = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
    ax.set_yscale("log")
    ax.set_ylim(1050, 100)
    ax.yaxis.set_major_locator(FixedLocator(major_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):d}"))
    ax.grid(False)


def plot_uwyo_skewt(
    ax: plt.Axes,
    sounding_df,
    *,
    skew: float = 35.0,
    xlim: tuple[float, float] = (-40.0, 60.0),
    barb_stride: int = 75,
) -> None:
    """
    Draw a Skew-T-like panel with thermodynamic grid and wind barbs.

    The background is intentionally styled to resemble the UWYO plot while
    remaining dependency-free.
    """
    profile = sounding_df.sort_values("pressure_hPa", ascending=False).copy()
    pressure = profile["pressure_hPa"].to_numpy(dtype=float)
    temperature = profile["temperature_C"].to_numpy(dtype=float)
    dewpoint = profile["dewpoint_C"].to_numpy(dtype=float)
    u_ms = profile["u_ms"].to_numpy(dtype=float)
    v_ms = profile["v_ms"].to_numpy(dtype=float)

    pressure_grid = np.geomspace(1000.0, 100.0, 180)
    _configure_pressure_axis(ax)
    ax.set_xlim(*xlim)
    ax.set_facecolor("white")

    for p in [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]:
        ax.axhline(p, color="0.75", lw=0.8, zorder=0)

    for temp_0 in np.arange(-40, 61, 10):
        ax.plot(
            _skew_x(np.full_like(pressure_grid, temp_0), pressure_grid, skew),
            pressure_grid,
            color="0.75",
            lw=0.8,
            zorder=0,
        )

    for theta_k in np.arange(260, 450, 10):
        dry_curve = _dry_adiabat_c(theta_k, pressure_grid)
        ax.plot(
            _skew_x(dry_curve, pressure_grid, skew),
            pressure_grid,
            color="#ff6b6b",
            lw=1.0,
            ls="--",
            alpha=0.9,
            zorder=0,
        )

    for start_temp_c in np.arange(-30, 46, 10):
        moist_curve = _moist_adiabat_c(start_temp_c, pressure_grid)
        ax.plot(
            _skew_x(moist_curve, pressure_grid, skew),
            pressure_grid,
            color="#6a7bff",
            lw=1.0,
            ls="--",
            alpha=0.9,
            zorder=0,
        )

    mixing_ratios = [0.4, 1, 2, 4, 7, 10, 16, 24, 32]
    for ratio in mixing_ratios:
        p_mix = np.geomspace(1000.0, 500.0, 80)
        td_mix = _dewpoint_from_mixing_ratio(p_mix, ratio)
        ax.plot(
            _skew_x(td_mix, p_mix, skew),
            p_mix,
            color="#4caf50",
            lw=1.0,
            ls="--",
            alpha=0.95,
            zorder=0,
        )
        label_x = _skew_x(td_mix[0], p_mix[0], skew)
        ax.text(label_x, 985, f"{ratio:.1f}", color="#43a047", fontsize=7, ha="center", va="bottom")

    ax.plot(_skew_x(temperature, pressure, skew), pressure, color="black", lw=1.8, zorder=4)
    ax.plot(_skew_x(dewpoint, pressure, skew), pressure, color="black", lw=1.8, zorder=4)

    height_labels = [
        (1000, "92"),
        (925, "586"),
        (850, "1268"),
        (700, "2817"),
        (500, "5380"),
        (400, "6969"),
        (300, "8912"),
        (250, "10097"),
        (200, "11557"),
        (150, "13447"),
        (100, "16093"),
    ]
    for pressure_label, altitude in height_labels:
        if pressure.min() <= pressure_label <= pressure.max():
            ax.text(xlim[0] + 3.0, pressure_label, altitude, color="0.45", fontsize=7, va="center")

    barb_pressure = pressure[::barb_stride]
    barb_x = np.full_like(barb_pressure, xlim[1] - 3.0)
    ms_to_kt = 1.94384
    ax.barbs(
        barb_x,
        barb_pressure,
        u_ms[::barb_stride] * ms_to_kt,
        v_ms[::barb_stride] * ms_to_kt,
        length=5.0,
        linewidth=0.5,
        barbcolor="black",
        flagcolor="black",
        zorder=5,
    )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Pressure (hPa)")


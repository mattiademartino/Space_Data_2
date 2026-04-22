"""
plot.py - Quick temperature overview for the main atmospheric payloads.

This helper script is intentionally lightweight and aligned with the repo
conventions used in main.ipynb:

- canonical loaders from utils.py
- shared matplotlib style from utils.py
- figures saved inside figures/cross_dataset/
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from utils import apply_plot_style, load_grasp, load_obama, load_vamos_science, save_figure


def main() -> None:
    apply_plot_style()

    grasp = load_grasp().iloc[1:].copy()
    vamos = load_vamos_science()
    obama = load_obama().copy()

    grasp["t_min"] = grasp["t_rel"] / 60.0
    vamos["t_min"] = vamos["t_rel"] / 60.0
    obama["t_min"] = (
        (obama["Time_s"] - obama["Time_s"].iloc[0]) / 60.0
        if "Time_s" in obama
        else obama.index / 60.0
    )

    mask1 = obama["first_temp_avg_C"].between(-20, 50)
    mask2 = obama["second_temp_avg_C"].between(-20, 50)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle("Temperature overview - GRASP, VAMOS, OBAMA", fontweight="bold")

    ax.plot(grasp["t_min"], grasp["temp_C"], lw=0.9, color="tab:red", alpha=0.85, label="GRASP")
    ax.plot(vamos["t_min"], vamos["temp_C"], lw=0.6, color="tab:blue", alpha=0.75, label="VAMOS")
    if mask1.any():
        ax.plot(
            obama.loc[mask1, "t_min"],
            obama.loc[mask1, "first_temp_avg_C"],
            "o-",
            ms=4,
            lw=1.2,
            color="tab:green",
            label="OBAMA sensor 1",
        )
    if mask2.any():
        ax.plot(
            obama.loc[mask2, "t_min"],
            obama.loc[mask2, "second_temp_avg_C"],
            "D--",
            ms=4,
            lw=1.0,
            color="teal",
            alpha=0.6,
            label="OBAMA sensor 2",
        )

    ax.set_xlabel("Relative time (min)")
    ax.set_ylabel("Temperature (degC)")
    ax.legend(fontsize=9)

    save_figure(fig, "cross_dataset/fig_temp_overview.png")
    if "agg" in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Multi-probe CanSat atmospheric data analysis. Six payloads descended over Switzerland on 2026-02-05; raw sensor data is compared against external references (UWYO radiosonde, ERA5 reanalysis, ISA standard atmosphere). The pipeline applies signal-processing techniques (Butterworth low-pass, Welch PSD, STFT) to separate physical atmospheric signals from sensor artefacts and parachute-oscillation noise.

## Running the notebook

```bash
# Execute main.ipynb in-place (local)
jupyter nbconvert --to notebook --execute --inplace main.ipynb

# Or launch interactively
jupyter lab
```

The notebook must remain runnable top-to-bottom both locally and on Google Colab. Never use absolute paths inside the notebook — always import paths from `utils.py`.

## Architecture

`utils.py` is the single source of truth for:
- Canonical paths (`ROOT_DIR`, `DATA_DIR`, `EXTERNAL_DATA_DIR`, `FIG_DIR`, and per-probe constants)
- Dataset loaders: `load_grasp()`, `load_vamos_science()`, `load_vamos_wind()`, `load_obama()`, `load_uwyo_sounding()`, `load_era5_profile()`, `load_scamsat_bundle()`
- ISA standard atmosphere helpers: `isa_alt()`, `isa_press()`, `isa_temp_c()`, `baro_altitude_agl()`
- Signal processing: `resample_uniform()`, `butter_lowpass()`, `detrend_linear()`, `compute_welch_psd()`
- Figure helpers: `save_figure()`, `apply_plot_style()`, `style_card_plot()`
- Skew-T diagram: `plot_uwyo_skewt()`

`main.ipynb` is the official analysis narrative. It loads data via `utils.py`, applies the pipeline, and saves all figures. Add new loaders/paths to `utils.py` before using them in the notebook.

`tentativi/` is for exploratory scripts and experiments — nothing there should be imported or depended on by `main.ipynb`.

`cursed.ipynb` downloads ERA5 data from the CDS API and saves the NetCDF to `data/external_dataset/`.

## Data layout

| Probe | Path |
|---|---|
| GRASP (Probe 4) | `data/Probe_4_GRASP/run_0261 - with graphs.csv` |
| VAMOS science (Probe 3) | `data/Probe_3_VAMOS/logs/science.csv` |
| VAMOS wind (Probe 3) | `data/Probe_3_VAMOS/logs/wind.csv` |
| OBAMA (Probe 5) | `data/Probe_5_OBAMA/OBAMA_data_decoded.xlsx` |
| ScamSat (Probe 6) | `data/Probe_6_ScamSat/scamsat_data_upload/002/` |
| UWYO radiosonde | `data/external_dataset/uwyo_06610_2026-02-05_12Z.csv` |
| ERA5 NetCDF | `data/external_dataset/era5_dubendorf_20260206.nc` |

Legacy paths in `CanSat data utili/` are kept for backwards compatibility only — do not use them as input.

## Key data-loading quirks

- **GRASP row 0** is a sensor-init artefact (pressure spike); skip it with `df.iloc[1:]` before analysis.
- **VAMOS CSV files** contain repeated header rows due to a firmware bug; the loaders strip them automatically.
- **VAMOS wind** has a mid-recording timestamp reset (µC reboot); the loader discards everything before the last reset.
- **ScamSat** data is stored as raw binary (`uint16`/`int16`/`float32` via `np.fromfile`); each channel has its own scale factor and time-span constant defined in `utils.py`.

## Figures

All figures must be saved via `utils.save_figure(fig, "<subfolder>/<name>.png")`. Never call `plt.savefig(...)` directly. The canonical subdirectories are:

| Subfolder | Content |
|---|---|
| `plot_raw/` | Raw time series and initial overviews |
| `data_quality/` | Artefact and anomaly checks |
| `flight_dynamics/` | Descent, wind, flight-phase windows |
| `cross_dataset/` | Cross-probe comparisons |
| `signal_processing/` | FFT, PSD, filters, PM analysis |
| `external_dataset/` | UWYO, ERA5, Skew-T plots |
| `scamsat/` | ScamSat-specific outputs |
| `exports/` | JSON metadata exports |

## Report structure (`Report/`)

The LaTeX report in `Report/main.tex` mirrors the analysis pipeline:

1. **Research Question** — mission overview, sensor sensitivity to external disturbances (parachute oscillation, payload rotation)
2. **Methodology** — spectral methods rationale (why FFT/Welch/STFT), atmospheric data analysis flow
3. **Results** — structured in two blocks:
   - *Spectral analysis*: low-pass filter → linear detrend → FFT/Welch PSD → artefact identification and removal
   - *Atmospheric analysis*: vertical profile, cross-dataset comparison, external dataset (UWYO/ERA5)
4. **Interpretation and Discussion** — physical interpretation of results (thermal inversion, PM boundary layer accumulation, CO₂ above background, wind phase-segmentation, absence/presence of pendular oscillation)

Figures used in the report live in `Report/Img/` and are manually copied from `figures/`. Do not auto-generate into `Report/Img/` from the notebook.

## Checklist before finishing any change

- Data still read from `data/`?
- External datasets still in `data/external_dataset/`?
- New figures saved to `figures/<correct-subfolder>/`?
- `main.ipynb` still runs top-to-bottom without errors?

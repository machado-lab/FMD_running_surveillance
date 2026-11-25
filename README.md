# FMD Risk Surveillance Pipeline

Farm-level Foot-and-Mouth Disease (FMD) risk scoring for Rio Grande do Sul (RS), Brazil.  
The pipeline integrates spatial, network, and environmental data.

---

> [!WARNING]
> **The data in `raw_data/` is fully synthetic (fictitious).**
> GPS coordinates, names, CPF/CNPJ, and any other property identifiers have been
> replaced with randomized values solely to demonstrate that the code
> runs — **it does not represent real surveillance data** and cannot be
> used to identify any individual. Public datasets (roads, borders, RS
> boundary, temperature/LST, municipality-level wild boar survey) are real.
> To regenerate the synthetic data: `python tools/generate_mock_raw_data.py`.


## Project structure

```
project_root/
├── pipeline/               # 9 sequential analysis scripts
│   ├── 1_load_data.py
│   ├── 2_1_javalis_spatial_distribution.py
│   ├── 2_get_temperature_rs_with_raster.py
│   ├── 3_0_download_full_roads.py    # one-time, run manually (see below)
│   ├── 3_1_road_animal_preassure.py
│   ├── 3_get_spatial_net_features.py
│   ├── 5_distance_to_border.py
│   ├── 7_combine_and_check_data.py
│   └── 8_fmd_risk_analysis.py
│
├── raw_data/               # Input data (read-only, not version-controlled)
│   ├── cadastro/           # Farm registry CSVs
│   ├── gtas/               # Animal movement records (by species)
│   │   ├── bovine_movs/
│   │   ├── bubaline_movs/
│   │   ├── ovi_movs/
│   │   ├── capri_movs/
│   │   └── suine_movs/
│   ├── javali_data/        # Wild boar (javali) survey Excel file
│   ├── roads/              # roads_rs_major.shp (reference), roads_rs_full.shp (used by the pipeline)
│   ├── osm/                # Cached Brazil OSM extract (built by 3_0_download_full_roads.py)
│   ├── rs_boundary_ibge/   # RS_UF_2024.shp (optional, for border clipping)
│   ├── border/             # Natural Earth border lines (auto-downloaded)
│   └── hydro/              # Natural Earth rivers (auto-downloaded)
│
├── intermediate/           # Cached intermediate files (auto-generated)
│   ├── cadastro.pkl
│   ├── movements2.pkl
│   ├── LST_raster/         # Monthly MODIS temperature GeoTIFFs
│   ├── road_pressure/      # Shapefile outputs from the routing step
│   ├── tmp_monthly_flows/  # Monthly movement aggregations
│   └── tmp_road_pressure_full/  # Monthly road-pressure GPKG layers
│
├── results/                # Final pipeline outputs
│   ├── fmd_risk_scored.csv         ← primary deliverable
│   ├── fmd_risk_scored.pkl
│   ├── top50_high_risk_farms.csv
│   ├── component_weights.csv
│   ├── risk_score_summary_stats.csv
│   ├── sensitivity_analysis.csv
│   ├── sensitivity_spearman_corr.csv
│   └── plot*.png                   ← diagnostic plots
│
├── credentials/            # OAuth / API secrets (NOT version-controlled)
│   └── client_secret_*.json
│
├── src/
│   └── paths.py            # Centralized path definitions
│
├── run_pipeline.py         # Main entry point
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd FMD_surveillace_codesource
```

### 2. Create and activate a Python environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pandas geopandas numpy scikit-learn networkx scipy \
            libpysal geobr matplotlib seaborn rasterio geemap \
            earthengine-api pydrive tqdm openpyxl
```

> **Note:** `geobr` fetches IBGE municipality boundaries on first use; an internet connection is required.

> **Note:** `3_0_download_full_roads.py` additionally needs `pyrosm`
> (`pip install pyrosm`), which is not required by any other step — see
> that script's header for why it may need to run under a separate Python
> environment.

---

## Credential setup (steps 2b and 3a only)

Steps 2b (`2_get_temperature_rs_with_raster.py`) and 3a (`3_1_road_animal_preassure.py`)  
require access to **Google Earth Engine** and **Google Drive**.

1. Authenticate with Earth Engine (once):
   ```bash
   earthengine authenticate
   ```
2. Download the OAuth2 client secret JSON from the  
   [Google Cloud Console](https://console.cloud.google.com/) and place it in:
   ```
   credentials/client_secret_<your-id>.json
   ```
   The pipeline automatically looks for `credentials/client_secret_*.json`.
   **No credentials are committed to git** (the `credentials/` folder is in `.gitignore`).
3. Set your Earth Engine project id via environment variable:
   ```bash
   export GEE_PROJECT=your-ee-project
   ```

---

## Running the pipeline

### One-time setup: full road network

`3_1_road_animal_preassure.py` and `3_get_spatial_net_features.py` route
against `raw_data/roads/roads_rs_full.shp` (every drivable OSM highway
class, not just major roads). Build it once before the first run:

```bash
python pipeline/3_0_download_full_roads.py
```

This is not part of `run_pipeline.py`'s automatic sequence — it downloads a
large OSM extract and may need a separate Python environment (see the
script's header).

### Full pipeline (steps 1 – 8)

```bash
python run_pipeline.py
```

### Resume from a specific step

```bash
python run_pipeline.py --from 7   # re-runs only combine + risk analysis
```

The runner uses a **fail-fast** strategy: if any step exits with a non-zero code,  
the pipeline stops immediately and reports the step number that failed.

---

## Pipeline steps

| # | Script | Description | Heavy? |
|---|--------|-------------|--------|
| 1 | `1_load_data.py` | Loads farm registry (cadastro) and GTAS movement CSVs | No |
| 2 | `2_1_javalis_spatial_distribution.py` | Spatial smoothing of wild boar (javali) risk (Queen spatial weights) | No |
| 3 | `2_get_temperature_rs_with_raster.py` | Downloads MODIS LST via Earth Engine + computes FMD survival probability | Yes (download) |
| 4 | `3_1_road_animal_preassure.py` | OSM road routing (full network) + farm exposure via buffer | **Very heavy** (cached) |
| 5 | `3_get_spatial_net_features.py` | Network analysis, farm density, species-weighted animal pressure, road distances. Accepts an optional subset of components to (re)run, e.g. `python pipeline/3_get_spatial_net_features.py farm_density animal_pressure` | Moderate |
| 6 | `5_distance_to_border.py` | Distance to international border (auto-download from Natural Earth) | No |
| 7 | `7_combine_and_check_data.py` | Consolidates all feature sets | No |
| 8 | `8_fmd_risk_analysis.py` | Computes composite risk scores + sensitivity analysis + plots | No |

---

## Expected outputs

After a successful run, `results/` will contain:

| File | Description |
|------|-------------|
| `fmd_risk_scored.csv` / `.pkl` | **Primary output** — every farm with both risk scores and 8 normalized components |
| `top50_high_risk_farms.csv` | Top 50 farms by epi-weighted score |
| `component_weights.csv` | Equal vs. epi weight per component, for reproducibility |
| `risk_score_summary_stats.csv` | Summary statistics for both scores and all normalized components |
| `sensitivity_analysis.csv` | Top-N rank stability per ±20% perturbed weight |
| `sensitivity_spearman_corr.csv` | Full Spearman correlation matrix across perturbed score variants |
| `plot1_risk_score_histograms.png` | Equal-weight vs. epi-weight score distributions |
| `plot2_equal_vs_epi_scores.png` | Equal-weight vs. epi-weight score scatter |
| `plot3_components_vs_epi_score.png` | Top-4 weighted components vs. epi score |
| `plot5_sensitivity_heatmap.png` | Sensitivity analysis correlation heatmap |
| `plot7_component_correlation.png` | Spearman correlation among normalized components |

---

## Caching behavior (step 4)

Step 4 (road animal pressure) builds a road-network graph and computes shortest paths  
for every farm pair — this can take several hours on the first run.  
Intermediate results are cached in `intermediate/`:

- `graph_rs_roads_full.pkl` — raw OSM graph (full road network)
- `graph_rs_roads_full_simplified.pkl` — contracted (degree-2) graph
- `routes_roads_full_cache.pkl` — computed shortest paths
- `tmp_monthly_flows/*.pkl` — monthly movement aggregations
- `tmp_road_pressure_full/*.gpkg` — monthly road-pressure layers

These are named `*_full*` so they are never silently reused if you also keep
a major-roads-only run around (`src/paths.py` still defines the
non-`_FULL` constants for reference/comparison).

On subsequent runs, cached files are reused automatically.  
To force full recomputation, delete the relevant cache files in `intermediate/`.

---

## Path system

All paths are defined in [`src/paths.py`](src/paths.py).  
No absolute paths appear in the pipeline scripts.  
Every path derives from `BASE_DIR = Path(__file__).resolve().parent.parent`  
(i.e., the project root), so the pipeline runs on any machine without edits.

---

## Notes

- The pipeline was developed for RS, Brazil, using Brazilian government data formats.
- The scientific logic (algorithms, weights, model architecture) is unchanged from the original research version.
- Steps 2b and 4 require internet on the first run (Earth Engine download; the Natural Earth border auto-download in step 6 also needs internet).

## LINCENCE

Copyright (c) 2026 machado-lab

All Rights Reserved.

No part of this software may be copied, modified, distributed, sublicensed, or commercially or otherwise used without prior written permission from the copyright holder.

Permission is granted only upon written approval by the copyright holder.

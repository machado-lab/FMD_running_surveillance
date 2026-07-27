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
├── pipeline/               # 8 sequential analysis scripts
│   ├── 1_load_data.py
│   ├── 2_1_javalis_spatial_distribution.py
│   ├── 2_get_temperature_rs_with_raster.py
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
│   ├── roads/              # roads_rs_major.shp
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
│   └── tmp_road_pressure/  # Monthly road-pressure GPKG layers
│
├── results/                # Final pipeline outputs
│   └── fmd_risk_scored.csv         ← primary deliverable
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
            libpysal geobr matplotlib rasterio geemap \
            earthengine-api pydrive tqdm openpyxl
```

> **Note:** `geobr` fetches IBGE municipality boundaries on first use; an internet connection is required.

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
| 4 | `3_1_road_animal_preassure.py` | OSM road routing + farm exposure via buffer | **Very heavy** (cached) |
| 5 | `3_get_spatial_net_features.py` | Network analysis, farm density, animal pressure, road distances | Moderate |
| 6 | `5_distance_to_border.py` | Distance to international border (auto-download from Natural Earth) | No |
| 7 | `7_combine_and_check_data.py` | Consolidates all feature sets | No |
| 8 | `8_fmd_risk_analysis.py` | Computes composite risk scores + sensitivity analysis + plots | No |

---

## Expected outputs

After a successful run, `results/` will contain:

| File | Description |
|------|-------------|
| `fmd_risk_scored.csv` | **Primary output** — every farm with both risk scores and 8 normalized components |

---

## Caching behavior (step 4)

Step 4 (road animal pressure) builds a road-network graph and computes shortest paths  
for every farm pair — this can take several hours on the first run.  
Intermediate results are cached in `intermediate/`:

- `graph_rs_roads.pkl` — raw OSM graph
- `graph_rs_roads_simplified.pkl` — contracted (degree-2) graph
- `routes_roads_cache.pkl` — computed shortest paths
- `tmp_monthly_flows/*.pkl` — monthly movement aggregations
- `tmp_road_pressure/*.gpkg` — monthly road-pressure layers

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

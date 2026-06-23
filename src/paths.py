"""
Centralized path definitions for the FMD surveillance pipeline.

All pipeline scripts import from this module — never hardcode paths elsewhere.

Directory layout (relative to project root):
    raw_data/       — original input data (read-only)
    intermediate/   — cached intermediate outputs (reused across steps)
    results/        — final pipeline outputs (fmd_risk_scored.csv, plots, etc.)
    credentials/    — API keys / OAuth secrets (not committed to git)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root

# ---------------------------------------------------------------------------
# Top-level directories
# ---------------------------------------------------------------------------
RAW_DATA        = BASE_DIR / "raw_data"
INTERMEDIATE    = BASE_DIR / "intermediate"
RESULTS         = BASE_DIR / "results"
CREDENTIALS_DIR = BASE_DIR / "credentials"

# ---------------------------------------------------------------------------
# Raw data sub-paths (inputs — never written by pipeline)
# ---------------------------------------------------------------------------
CADASTRO_DIR    = RAW_DATA / "cadastro"
GTAS_DIR        = RAW_DATA / "gtas"
JAVALI_DIR      = RAW_DATA / "javali_data"
ROADS_SHP       = RAW_DATA / "roads" / "roads_rs_major.shp"
SISBRAVET_DIR   = RAW_DATA / "sisbravet"
EPICOLLECT_CSV  = RAW_DATA / "PNEFA_epicollec_bio_records" / "form-1__form-pvbr.csv"
RS_BOUNDARY_SHP = RAW_DATA / "rs_boundary_ibge" / "RS_UF_2024.shp"
BORDER_DIR      = RAW_DATA / "border"
HYDRO_DIR       = RAW_DATA / "hydro"
LST_DIR         = INTERMEDIATE / "LST_raster"

# ---------------------------------------------------------------------------
# Intermediate pickle / shapefile paths (produced and consumed by pipeline)
# ---------------------------------------------------------------------------
CADASTRO_PKL            = INTERMEDIATE / "cadastro.pkl"
MOVEMENTS_PKL           = INTERMEDIATE / "movements2.pkl"
JAVALI_PKL              = INTERMEDIATE / "javali_risk_smoothed.pkl"
WEATHER_PKL             = INTERMEDIATE / "weather_probs_fmd.pkl"
NETWORK_PKL             = INTERMEDIATE / "network_farms_rank.pkl"
FARM_DENSITY_PKL        = INTERMEDIATE / "farm_density_pressure.pkl"
ANIMAL_PRESSURE_PKL     = INTERMEDIATE / "animal_pressure_within_radius.pkl"
DISTANCE_ROADS_PKL      = INTERMEDIATE / "distance_to_roads.pkl"
SLAUGHTERHOUSE_PKL      = INTERMEDIATE / "slaughterhouse_proximity_risk.pkl"
SISBRAVET_PKL           = INTERMEDIATE / "sisbravet_fmd_combined.pkl"
BIOSEC_PKL              = INTERMEDIATE / "cadastro_with_risk.pkl"
BORDER_PKL              = INTERMEDIATE / "distance_to_border.pkl"
BANCO_FINAL_PKL         = INTERMEDIATE / "banco_final_to_risk_analysis.pkl"

# Road pressure sub-tree (heavy spatial caches)
GRAPH_RAW_CACHE         = INTERMEDIATE / "graph_rs_roads.pkl"
GRAPH_SIMP_CACHE        = INTERMEDIATE / "graph_rs_roads_simplified.pkl"
ROUTE_CACHE             = INTERMEDIATE / "routes_roads_cache.pkl"
TMP_MONTHLY             = INTERMEDIATE / "tmp_monthly_flows"
TMP_PRESSURE            = INTERMEDIATE / "tmp_road_pressure"
ROAD_OUT                = INTERMEDIATE / "road_pressure"
FARM_ROAD_PRESSURE_SHP  = ROAD_OUT / "farm_road_pressure_10km.shp"

# Intermediate model/analysis outputs from step 6
R_PLOTS_DIR             = INTERMEDIATE / "r_plots"


def ensure_dirs() -> None:
    """Create all output directories that must exist before the pipeline runs."""
    for d in [
        RESULTS, INTERMEDIATE, LST_DIR,
        TMP_MONTHLY, TMP_PRESSURE, ROAD_OUT,
        R_PLOTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


# Create directories immediately on import so every script can write freely.
ensure_dirs()

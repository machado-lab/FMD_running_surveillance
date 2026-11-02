######################################################################
#           combine and check each dataset                           #
######################################################################
# load packages
import sys
import os
import pickle
from pathlib import Path

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import numpy as np
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import (
    INTERMEDIATE,
    CADASTRO_PKL, FARM_DENSITY_PKL, ANIMAL_PRESSURE_PKL,
    DISTANCE_ROADS_PKL, WEATHER_PKL, BORDER_PKL,
    NETWORK_PKL,
    JAVALI_PKL, SLAUGHTERHOUSE_PKL, BANCO_FINAL_PKL,
    FARM_ROAD_PRESSURE_SHP,
)

#================================================================#
#  cadastro ----
#================================================================#
with open(str(CADASTRO_PKL), "rb") as f:
    cadastro = pickle.load(f)

print("Cadastro shape:", cadastro.shape)
print("Cadastro columns:", cadastro.columns.tolist()[:10])
print("\nCadastro sample:\n", cadastro.head())

#================================================================#
#  Load spatial features (network, farm pressure, animal pressure, roads)
#================================================================#

#  Farm density pressure
with open(str(FARM_DENSITY_PKL), "rb") as f:
    farm_pressure_df = pickle.load(f)


#  Animal pressure
with open(str(ANIMAL_PRESSURE_PKL), "rb") as f:
    animal_pressure_df = pickle.load(f)

#  Distance to roads
with open(str(DISTANCE_ROADS_PKL), "rb") as f:
    roads_df = pickle.load(f)

#  Road animal pressure (precomputed — do NOT recompute 3_1_road_animal_preassure.py)
road_pressure_path = str(FARM_ROAD_PRESSURE_SHP)
assert os.path.exists(road_pressure_path), f"Road pressure file not found: {road_pressure_path}"
road_pressure_gdf = gpd.read_file(road_pressure_path)
road_pressure_gdf = road_pressure_gdf.rename(columns={"COD_PROPRI": "COD_PROPRIEDADE_RURAL"})
road_pressure_gdf["COD_PROPRIEDADE_RURAL"] = (
    pd.to_numeric(road_pressure_gdf["COD_PROPRIEDADE_RURAL"], errors="coerce").astype("Int64")
)
assert "total_pres" in road_pressure_gdf.columns, "Missing column: total_pres in road pressure shapefile"
assert road_pressure_gdf["COD_PROPRIEDADE_RURAL"].duplicated().sum() == 0, \
    "Duplicate farm IDs in road pressure shapefile"
n_nan_pres  = road_pressure_gdf["total_pres"].isna().sum()
n_zero_pres = (road_pressure_gdf["total_pres"] == 0).sum()
print(f"\nRoad pressure validation — shape: {road_pressure_gdf.shape}, "
      f"NaN total_pres: {n_nan_pres}, "
      f"zero total_pres (no nearby traffic, legitimate): {n_zero_pres}")
print(road_pressure_gdf["total_pres"].describe().round(2))
road_pressure_gdf = road_pressure_gdf[["COD_PROPRIEDADE_RURAL", "total_pres"]].copy()

#  Load weather probability data
with open(str(WEATHER_PKL), "rb") as f:
    weather_df = pickle.load(f)

# distance to the border
with open(str(BORDER_PKL), "rb") as f:
    borders_df = pickle.load(f)

# network
with open(str(NETWORK_PKL), "rb") as f:
    network_df = pickle.load(f)

# javalies
with open(str(JAVALI_PKL), "rb") as f:
    javalies_df = pickle.load(f)
# distance to slaughterhouses
with open(str(SLAUGHTERHOUSE_PKL), "rb") as f:
    abate_distance = pickle.load(f)


#================================================================#
#  Collapse cadastro to one row per farm.
#
#  cadastro.pkl is long-format (one row per farm-species combination —
#  restored by 1_load_data.py's fix to the per-species collapse bug).
#  Every other farm-level feature merged below (network, farm/animal
#  pressure, roads, weather, borders, javali, slaughterhouse) expects
#  exactly one row per COD_PROPRIEDADE_RURAL, so collapse here, before any
#  of those merges, rather than relying on drop_duplicates() further down
#  to paper over a many-rows-per-farm table.
#================================================================#
_id = "COD_PROPRIEDADE_RURAL"
_first_cols = [c for c in cadastro.columns if c not in ("ESPECIE_ANIMAL", "QTD_DECLARADA", _id)]

_nunique_per_farm = cadastro.groupby(_id)[_first_cols].nunique(dropna=False)
_mismatched = (_nunique_per_farm > 1).any(axis=1)
if _mismatched.any():
    print(f"WARNING: {_mismatched.sum()} farms have inconsistent values across "
          f"their species rows for one or more of {_first_cols} — using the "
          f"first non-null value per farm (likely minor geocoding drift "
          f"between declarations, not expected to be common).")

cadastro_farm = cadastro.groupby(_id, as_index=False)[_first_cols].first()
cadastro_farm["QTD_DECLARADA_total"] = (
    cadastro.groupby(_id)["QTD_DECLARADA"].sum().values
)
cadastro = cadastro_farm
assert not cadastro[_id].duplicated().any(), "cadastro must be one row per farm after collapse"

#================================================================#
#  Combine all datasets
#================================================================#
combined_df = cadastro.copy()
n_initial = combined_df["COD_PROPRIEDADE_RURAL"].nunique()
print(f"Initial unique farms: {n_initial}, total rows: {len(combined_df)}")

# Merge network features 1
network_df = network_df[["farm_id", "network_risk_score"]].copy()
network_df["farm_id"] = pd.to_numeric(network_df["farm_id"], errors='coerce').astype("Int64")
network_df = network_df.drop_duplicates(subset=["farm_id"])

combined_df = combined_df.merge(
  network_df,
  left_on = "COD_PROPRIEDADE_RURAL",
  right_on = "farm_id",
  how = "left"
)
combined_df = combined_df.drop(columns=["farm_id"])

# Merge farm pressure
farm_pressure_df = farm_pressure_df[["COD_PROPRIEDADE_RURAL", "n_neighbors_5km"]].copy()
farm_pressure_df = farm_pressure_df.drop_duplicates(subset=["COD_PROPRIEDADE_RURAL"])
combined_df = combined_df.merge(farm_pressure_df, on=['COD_PROPRIEDADE_RURAL'], how='left', suffixes=('', '_farm'))

# Merge animal pressure — animal_pressure_within_radius.pkl is already one
# row per farm (species-weighted cross-species sum computed in
# pipeline/3_get_spatial_net_features.py), so this is a plain 1:1 merge;
# assert it, rather than silently relying on drop_duplicates to paper over
# a many-rows-per-farm table.
animal_pressure_df = animal_pressure_df[["COD_PROPRIEDADE_RURAL", "animal_pressure_weighted_10km"]].copy()
assert not animal_pressure_df["COD_PROPRIEDADE_RURAL"].duplicated().any(), \
    "animal pressure output must be one row per farm"
combined_df = combined_df.merge(animal_pressure_df, on="COD_PROPRIEDADE_RURAL", how="left")

# Merge roads — keep raw sum-distance for road_exposure_risk_raw computation
roads_df = roads_df[["COD_PROPRIEDADE_RURAL", "dist_3_roads_sum_m"]].copy()
roads_df = roads_df.drop_duplicates(subset=["COD_PROPRIEDADE_RURAL"])
combined_df = combined_df.merge(roads_df, on=['COD_PROPRIEDADE_RURAL'], how='left')

# Merge weather
weather_df = weather_df[["COD_PROPRIEDADE_RURAL", "fmd_survival_prob_overall"]].copy()  # RS only wants a single value not monthly forecast
weather_df = weather_df.drop_duplicates(subset=["COD_PROPRIEDADE_RURAL"])
combined_df = combined_df.merge(weather_df, on='COD_PROPRIEDADE_RURAL', how='left')

# Merge distance from the border
borders_df = borders_df[["COD_PROPRIEDADE_RURAL", "dist_border_m", "nearest_border_type"]].copy()
borders_df = borders_df.drop_duplicates(subset=["COD_PROPRIEDADE_RURAL"])
combined_df = combined_df.merge(borders_df, on='COD_PROPRIEDADE_RURAL', how='left')

# Merge javali risk (municipality-level)
combined_df["COD_MUNICIPIO_IBGE"] = combined_df["COD_MUNICIPIO_IBGE"].astype(str)
javalies_df["code_muni"] = javalies_df["code_muni"].astype(str)
javalies_df = javalies_df[["code_muni", "javali_risk_smooth_iter"]].copy()
javalies_df = javalies_df.drop_duplicates(subset=["code_muni"])
combined_df = combined_df.merge(javalies_df, left_on="COD_MUNICIPIO_IBGE", right_on="code_muni", how="left")
combined_df = combined_df.drop("code_muni", axis = 1)
print(f"After all merges — unique farms: {combined_df['COD_PROPRIEDADE_RURAL'].nunique()}, total rows: {len(combined_df)}")

#merge distance to slaughterhouses
abate_distance.columns
abate_distance = abate_distance[["COD_PROPRIEDADE_RURAL", "slaughterhouse_risk"]].copy()
abate_distance = abate_distance.drop_duplicates(subset=["COD_PROPRIEDADE_RURAL"])
combined_df = combined_df.merge(abate_distance, left_on = "COD_PROPRIEDADE_RURAL", right_on = "COD_PROPRIEDADE_RURAL", how = "left" )

# Merge road animal pressure
road_pressure_gdf = road_pressure_gdf.drop_duplicates(subset=["COD_PROPRIEDADE_RURAL"])
combined_df = combined_df.merge(road_pressure_gdf, on="COD_PROPRIEDADE_RURAL", how="left")
n_nan_pres_post = combined_df["total_pres"].isna().sum()
print(f"\nAfter merging road pressure — NaN total_pres: {n_nan_pres_post} "
      f"(farms absent from road pressure shapefile — will receive median imputation)")

# ── ROAD EXPOSURE RISK COMPUTATION ───────────────────────────────────────────
# FMD spatial transmission kernel: >50% of transmission events occur within
# 3 km of a source; >80% within 10 km  λ is set so exposure halves at a sum-distance of 3,000 m (≈ 1 km per
# road), anchoring directly to the 3 km primary threshold. At 10 km
# sum-distance (≈ 3.3 km per road), residual exposure drops to ~10%,
# consistent with the >80% attenuation by 10 km.(Geographic and topographic determinants of local FMD transmission applied to the 2001 UK FMD epidemic)
#
# Numerical check — exp(−λ × dist_sum_m):
#   1 km → 0.79  |  3 km → 0.50  |  10 km → 0.10  |  20 km → 0.01
HALF_LIFE_M  = 3_000                      # 3 km: >50% transmission threshold
DECAY_LAMBDA = np.log(2) / HALF_LIFE_M   # ≈ 2.310 × 10⁻⁴ per metre

# Impute NaN before computation
if combined_df["total_pres"].isna().sum() > 0:
    med_pres = combined_df["total_pres"].median()
    combined_df["total_pres"] = combined_df["total_pres"].fillna(med_pres)
    print(f"  total_pres imputed with median = {med_pres:.2f}")

if combined_df["dist_3_roads_sum_m"].isna().sum() > 0:
    med_dist = combined_df["dist_3_roads_sum_m"].median()
    combined_df["dist_3_roads_sum_m"] = combined_df["dist_3_roads_sum_m"].fillna(med_dist)
    print(f"  dist_3_roads_sum_m imputed with median = {med_dist:.2f}")

# Cap extreme distances at 99.9th percentile (245 farms exceed 100 km —
# likely geocoding artifacts; capping prevents spurious near-zero decay)
dist_cap_m  = combined_df["dist_3_roads_sum_m"].quantile(0.999)
dist_capped = combined_df["dist_3_roads_sum_m"].clip(upper=dist_cap_m)
n_capped    = (combined_df["dist_3_roads_sum_m"] > dist_cap_m).sum()
print(f"  Distance capped at {dist_cap_m:.0f} m (99.9th pct) — {n_capped} farms affected")

# R_road = log(1 + P_road) × exp(−λ × D_road)
# log1p reduces influence of extreme flow values in total_pres
# exp decay ensures risk strictly decreases with distance from roads
road_raw = np.log1p(combined_df["total_pres"]) * np.exp(-DECAY_LAMBDA * dist_capped)
road_raw = road_raw.replace([np.inf, -np.inf], 0)
combined_df["road_exposure_risk_raw"] = road_raw

assert combined_df["road_exposure_risk_raw"].isna().sum() == 0, "NaN in road_exposure_risk_raw"
assert not np.isinf(combined_df["road_exposure_risk_raw"]).any(), "Inf in road_exposure_risk_raw"
print(f"\nRoad exposure risk (raw) summary:")
print(combined_df["road_exposure_risk_raw"].describe().round(4))

# Drop intermediate columns — both captured in road_exposure_risk_raw
combined_df = combined_df.drop(columns=["total_pres", "dist_3_roads_sum_m"])

#================================================================#
#  Filtering by FMD susceptible species
#================================================================#
# dominant_species (set in pipeline/1_load_data.py) is already drawn only
# from FMD-susceptible species — this is now a defensive check, not an
# operative row filter: it should drop 0 rows by construction.
FMD_SPECIES = ["Bovinos", "Bubalinos", "Ovino", "Suíno", "Caprinos"]
combined_df["has_fmd_species"] = combined_df["dominant_species"].isin(FMD_SPECIES)
n_dropped = (~combined_df["has_fmd_species"]).sum()
if n_dropped:
    print(f"WARNING: dropping {n_dropped} farms with a dominant_species outside "
          f"{FMD_SPECIES} — this should not happen if 1_load_data.py's FMD_SPECIES "
          f"filter ran correctly upstream.")
combined_df = combined_df[combined_df["has_fmd_species"]].copy()
combined_df = combined_df.drop(columns=["has_fmd_species"])
print("Combined dataset shape after filtering for FMD susceptible species:", combined_df.shape)


#================================================================#
#  Check combined data
#================================================================#
missing = combined_df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing
print(combined_df.describe())

#================================================================#
#  Save combined dataset
#================================================================#
with open(str(BANCO_FINAL_PKL), 'wb') as f:
    pickle.dump(combined_df, f)
print(f"\nSaved: {BANCO_FINAL_PKL}")

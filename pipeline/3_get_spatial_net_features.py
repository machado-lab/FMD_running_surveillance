import os
import sys
import glob
import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import rasterio
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import BallTree
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import (
    CADASTRO_PKL, MOVEMENTS_PKL, ROADS_SHP, INTERMEDIATE,
    NETWORK_PKL, FARM_DENSITY_PKL, ANIMAL_PRESSURE_PKL,
    DISTANCE_ROADS_PKL, SLAUGHTERHOUSE_PKL,
)

# Load data from pickles
with open(str(CADASTRO_PKL), 'rb') as file:
    farms = pickle.load(file)


# Convert farms to GeoDataFrame
def farms_to_gdf(farms, crs="EPSG:4326"):
    return gpd.GeoDataFrame(
        farms.copy(),
        geometry=gpd.points_from_xy(farms["LONGITUDE_DECIMAL"], farms["LATITUDE_DECIMAL"]),
        crs=crs,
    )

gdf = farms_to_gdf(farms)
#Network analysis from GTAs
def build_movement_network(file_path, out_pkl=None):
    with open(str(file_path), 'rb') as file:
        gtas = pickle.load(file)

    # drop slaughterhouses for the network analysis
    gtas.columns
    gtas["FINALIDADE"].unique()
    # remove abate from the movemets
    gtas = gtas[~gtas["FINALIDADE"].isin(["Abate", 'Abate San.'])]


    df = gtas[
        [
            "DATA_EMISSAO",
            "COD_PROPRIEDADE_ORIGEM",
            "COD_PROPRIEDADE_DESTINO",
            "animal_total",
        ]
    ].dropna()

    def normalize_farm_id(series):
        s = series.astype(str).str.strip()
        s = s.str.replace(r"\\.0$", "", regex=True)
        s = s.str.replace(r"\\D", "", regex=True)
        s = s.replace("", np.nan)
        return s

    df["COD_PROPRIEDADE_ORIGEM"] = normalize_farm_id(df["COD_PROPRIEDADE_ORIGEM"])
    df["COD_PROPRIEDADE_DESTINO"] = normalize_farm_id(df["COD_PROPRIEDADE_DESTINO"])
    df["animal_total"] = pd.to_numeric(df["animal_total"], errors="coerce")

    df = df.dropna(subset=["COD_PROPRIEDADE_ORIGEM", "COD_PROPRIEDADE_DESTINO", "animal_total"])

    edges = (
        df.groupby(
            ["COD_PROPRIEDADE_ORIGEM", "COD_PROPRIEDADE_DESTINO"],
            as_index=False,
        )
        .agg(
            n_moves=("animal_total", "size"),
            weight=("animal_total", "sum"),
        )
    )

    G = nx.DiGraph()
    for _, r in edges.iterrows():
        G.add_edge(
            r["COD_PROPRIEDADE_ORIGEM"],
            r["COD_PROPRIEDADE_DESTINO"],
            weight=r["weight"],
            n_moves=r["n_moves"],
        )

    nodes = list(G.nodes())
    degree_df = pd.DataFrame({
        "farm_id": nodes,
        "in_degree": [G.in_degree(n) for n in nodes],
        "out_degree": [G.out_degree(n) for n in nodes],
        "total_degree": [G.degree(n) for n in nodes],
        "in_degree_weighted": [G.in_degree(n, weight="weight") for n in nodes],
        "out_degree_weighted": [G.out_degree(n, weight="weight") for n in nodes],
        "total_degree_weighted": [G.degree(n, weight="weight") for n in nodes],
    })

    scaler = MinMaxScaler()
    score_vars = [
        "in_degree",
        "out_degree",
        "in_degree_weighted",
        "out_degree_weighted",
        "total_degree"
    ]
    degree_df[score_vars] = scaler.fit_transform(degree_df[score_vars])

    degree_df["network_risk_score"] = (
        #0.4 * degree_df["in_degree"]
        + 0.4 * degree_df["in_degree_weighted"]
        #+ 0.1 * degree_df["out_degree"]
        + 0.1 * degree_df["out_degree_weighted"]
        + 0.5 * degree_df["total_degree"]

    )

    if out_pkl:
        degree_df.to_pickle(str(out_pkl))

    return degree_df

network = build_movement_network(
    MOVEMENTS_PKL,
    out_pkl=NETWORK_PKL,
)


#Farm density within radius
def farm_density_pressure(gdf, radius_m=10000, out_pkl=None):
    # drop rows without geometry or with invalid coords
    gdf = gdf.dropna(subset=["geometry"]).copy()
    gdf_m = gdf.to_crs(epsg=3857)
    gdf_m = gdf_m[np.isfinite(gdf_m.geometry.x) & np.isfinite(gdf_m.geometry.y)].copy()

    coords = np.vstack([gdf_m.geometry.x, gdf_m.geometry.y]).T
    tree = BallTree(coords, metric="euclidean")

    neighbors = tree.query_radius(coords, r=radius_m)
    counts = np.array([len(n) for n in neighbors]) - 1

    gdf_m["n_neighbors_5km"] = counts
    gdf_m["pressure_rank"] = (
        gdf_m["n_neighbors_5km"].rank(ascending=False, method="dense").astype(int)
    )

    x = gdf_m["n_neighbors_5km"].values.astype(float)
    gdf_m["n_neighbors_5km_normalized"] = (x - x.min()) / (x.max() - x.min())

    if out_pkl:
        gdf_m.to_pickle(str(out_pkl))

    return gdf_m

farm_density_pressuredf = farm_density_pressure(
    gdf=gdf,
    radius_m=10000,
    out_pkl=FARM_DENSITY_PKL,
)

# animal desnisty presuare ---
def _animal_pressure_one_species(
    sp,
    sp_gdf,
    radius_m,
    chunk_size,
    use_float32,
    tile_size_m,
    capacity_col,
):
    if sp_gdf.empty:
        return sp, sp_gdf

    sp_gdf = sp_gdf.reset_index(drop=True)
    cap_all = sp_gdf[capacity_col].values
    sp_gdf["tile_x"] = np.floor(sp_gdf["x"] / tile_size_m).astype(int)
    sp_gdf["tile_y"] = np.floor(sp_gdf["y"] / tile_size_m).astype(int)

    pressure = np.zeros(len(sp_gdf), dtype=float)

    for (_tx, _ty), tile_core in sp_gdf.groupby(["tile_x", "tile_y"]):
        if tile_core.empty:
            continue
        minx = tile_core["x"].min() - radius_m
        maxx = tile_core["x"].max() + radius_m
        miny = tile_core["y"].min() - radius_m
        maxy = tile_core["y"].max() + radius_m

        candidates = sp_gdf[
            (sp_gdf["x"] >= minx)
            & (sp_gdf["x"] <= maxx)
            & (sp_gdf["y"] >= miny)
            & (sp_gdf["y"] <= maxy)
        ]
        if candidates.empty:
            continue

        cand_coords = candidates[["x", "y"]].values
        if use_float32:
            cand_coords = cand_coords.astype("float32")
        tree = BallTree(cand_coords, metric="euclidean")
        cand_capacity = candidates[capacity_col].values

        core_idx = tile_core.index.to_numpy()
        core_coords = sp_gdf.loc[core_idx, ["x", "y"]].values
        if use_float32:
            core_coords = core_coords.astype("float32")

        for start in range(0, len(core_coords), chunk_size):
            end = min(start + chunk_size, len(core_coords))
            neighbors = tree.query_radius(core_coords[start:end], r=radius_m)
            for j, idx in enumerate(neighbors):
                core_pos = core_idx[start + j]
                pressure[core_pos] = cand_capacity[idx].sum() - cap_all[core_pos]

    sp_gdf["birds_within_5km"] = pressure
    x = pressure
    sp_gdf["birds_pressure_normalized"] = (
        (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
    )
    sp_gdf = sp_gdf.drop(columns=["x", "y", "tile_x", "tile_y"])
    return sp, sp_gdf


#animal capacity pressure (by species)
def animal_pressure_within_radius(
    gdf_m,
    radius_m=10000,
    species_col="ESPECIE_ANIMAL",
    capacity_col="QTD_DECLARADA",
    id_col="COD_PROPRIEDADE_RURAL",
    out_dir=None,
    chunk_size=2000,
    use_float32=True,
    epsg_m=3857,
    tile_size_m=20000,
    keep_cols=None,
    parallel=True,
    max_workers=4,
    return_combined=True,
):
    # project to metric CRS if needed
    if gdf_m.crs is None or gdf_m.crs.to_epsg() != epsg_m:
        gdf_m = gdf_m.to_crs(epsg=epsg_m)

    if keep_cols is None:
        keep_cols = [species_col, capacity_col, "geometry"]
        if id_col in gdf_m.columns:
            keep_cols.append(id_col)
    gdf_m = gdf_m[keep_cols].copy()

    # drop rows without geometry or without species/capacity
    gdf_m = gdf_m.dropna(subset=["geometry", species_col, capacity_col]).copy()
    gdf_m["x"] = gdf_m.geometry.x
    gdf_m["y"] = gdf_m.geometry.y
    gdf_m = gdf_m[np.isfinite(gdf_m["x"]) & np.isfinite(gdf_m["y"])].copy()
    gdf_m[capacity_col] = pd.to_numeric(gdf_m[capacity_col], errors="coerce").fillna(0)

    results = []
    groups = list(gdf_m.groupby(species_col))
    # Use threads, not processes: BallTree construction and query_radius release
    # the GIL, so threads parallelize the heavy work without pickling large
    # GeoDataFrames to workers — and without the macOS spawn / native-lib crashes
    # (BrokenProcessPool) that ProcessPoolExecutor caused here. This matches the
    # thread backend already used for the network/density features above.
    if parallel and max_workers > 1 and len(groups) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for sp, sp_gdf in groups:
                futures.append(
                    executor.submit(
                        _animal_pressure_one_species,
                        sp,
                        sp_gdf,
                        radius_m,
                        chunk_size,
                        use_float32,
                        tile_size_m,
                        capacity_col,
                    )
                )
            for fut in as_completed(futures):
                sp, sp_gdf = fut.result()
                if sp_gdf is None or sp_gdf.empty:
                    continue
                if out_dir:
                    sp_safe = (
                        str(sp)
                        .strip()
                        .replace(" ", "_")
                        .replace("/", "_")
                        .replace("\\", "_")
                    )
                    out_path = Path(out_dir) / f"animal_pressure_within_radius_{sp_safe}.pkl"
                    sp_gdf.to_pickle(str(out_path))
                if return_combined:
                    results.append(sp_gdf)
    else:
        for sp, sp_gdf in groups:
            sp, sp_gdf = _animal_pressure_one_species(
                sp,
                sp_gdf,
                radius_m,
                chunk_size,
                use_float32,
                tile_size_m,
                capacity_col,
            )
            if sp_gdf is None or sp_gdf.empty:
                continue
            if out_dir:
                sp_safe = (
                    str(sp)
                    .strip()
                    .replace(" ", "_")
                    .replace("/", "_")
                    .replace("\\", "_")
                )
                out_path = Path(out_dir) / f"animal_pressure_within_radius_{sp_safe}.pkl"
                sp_gdf.to_pickle(str(out_path))
            if return_combined:
                results.append(sp_gdf)

    if not results:
        return gdf_m.iloc[0:0].copy()

    return pd.concat(results, ignore_index=True)

if __name__ == "__main__":
    animal_pressure_within_radius_df = animal_pressure_within_radius(
        gdf_m=gdf,
        radius_m=10000,
        species_col="ESPECIE_ANIMAL",
        capacity_col="QTD_DECLARADA",
        id_col="COD_PROPRIEDADE_RURAL",
        out_dir=str(INTERMEDIATE),
        chunk_size=1000,
        tile_size_m=2000,   # adjust if needed
        epsg_m=3857,         # metric CRS
        parallel=True,
        max_workers=4,
        return_combined=True,
    )

    # Save combined animal pressure
    animal_pressure_within_radius_df.to_pickle(str(ANIMAL_PRESSURE_PKL))



############################################
#Distance to major roads
############################################
def distance_to_roads(gdf_farms, roads_shp, epsg_m=31982, out_pkl=None):
    return distance_to_roads_efficient(
        gdf_farms=gdf_farms,
        roads_shp=roads_shp,
        epsg_m=epsg_m,
        out_pkl=out_pkl,
    )


def _nearest_k_distances(roads_gdf, sindex, geom, k=3):
    # Try spatial index nearest (fast). Fallback to brute force if unavailable.
    try:
        idx = list(sindex.nearest(geom, num_results=k))
        dists = roads_gdf.geometry.iloc[idx].distance(geom).values
        return np.sort(dists)[:k]
    except TypeError:
        dists = roads_gdf.geometry.distance(geom)
        return np.sort(dists.nsmallest(k).values)


def _distance_chunk(farms_chunk, roads_gdf, k=3):
    sindex = roads_gdf.sindex
    dist1 = np.zeros(len(farms_chunk))
    dist2 = np.zeros(len(farms_chunk))
    dist3 = np.zeros(len(farms_chunk))

    for i, geom in enumerate(farms_chunk.geometry):
        if geom is None or geom.is_empty:
            dist1[i] = np.nan
            dist2[i] = np.nan
            dist3[i] = np.nan
            continue
        d = _nearest_k_distances(roads_gdf, sindex, geom, k=k)
        if len(d) < 3:
            d = np.pad(d, (0, 3 - len(d)), constant_values=np.nan)
        dist1[i], dist2[i], dist3[i] = d[:3]

    return dist1, dist2, dist3


def distance_to_roads_efficient(
    gdf_farms,
    roads_shp,
    epsg_m=31982,
    out_pkl=None,
    k=3,
    chunk_size=2000,
    parallel=True,
    max_workers=4,
    keep_cols=None,
    parallel_backend="process",
):
    roads = gpd.read_file(str(roads_shp)).to_crs(epsg=epsg_m)
    roads = roads[["geometry"]].dropna().copy()

    farms = gdf_farms.to_crs(epsg=epsg_m)
    if keep_cols is not None:
        keep_cols = list(set(keep_cols + ["geometry"]))
        farms = farms[keep_cols].copy()
    farms = farms.dropna(subset=["geometry"]).copy()

    # process in chunks (optionally parallel)
    dist1_all = np.zeros(len(farms))
    dist2_all = np.zeros(len(farms))
    dist3_all = np.zeros(len(farms))

    indices = list(range(0, len(farms), chunk_size))
    is_interactive = not hasattr(sys.modules.get("__main__"), "__file__")
    if parallel and parallel_backend == "process" and is_interactive:
        # Avoid multiprocessing spawn issues in interactive sessions
        parallel_backend = "thread"

    if parallel and max_workers > 1 and len(indices) > 1:
        Executor = ThreadPoolExecutor if parallel_backend == "thread" else ProcessPoolExecutor
        with Executor(max_workers=max_workers) as executor:
            futures = {}
            for start in indices:
                end = min(start + chunk_size, len(farms))
                chunk = farms.iloc[start:end].copy()
                futures[executor.submit(_distance_chunk, chunk, roads, k)] = (start, end)
            for fut in as_completed(futures):
                start, end = futures[fut]
                d1, d2, d3 = fut.result()
                dist1_all[start:end] = d1
                dist2_all[start:end] = d2
                dist3_all[start:end] = d3
    else:
        for start in indices:
            end = min(start + chunk_size, len(farms))
            chunk = farms.iloc[start:end].copy()
            d1, d2, d3 = _distance_chunk(chunk, roads, k)
            dist1_all[start:end] = d1
            dist2_all[start:end] = d2
            dist3_all[start:end] = d3

    farms["dist_road_1_m"] = dist1_all
    farms["dist_road_2_m"] = dist2_all
    farms["dist_road_3_m"] = dist3_all
    farms["dist_3_roads_sum_m"] = dist1_all + dist2_all + dist3_all

    if out_pkl:
        farms.to_pickle(str(out_pkl))

    return farms

distance_to_roadsdf = distance_to_roads_efficient(
    gdf_farms=gdf,
    roads_shp=ROADS_SHP,
    epsg_m=31982,
    out_pkl=DISTANCE_ROADS_PKL,
    parallel=True,
    max_workers=4,
    parallel_backend="thread",  # safe in interactive
)

# ============================================
# SLAUGHTERHOUSE PROXIMITY RISK
# ============================================
EARTH_RADIUS_KM = 6371.0
SLAUGHTER_CUTOFF_KM = 5.0


def slaughterhouse_proximity_risk(
    cadastro_pkl,
    movements_pkl,
    out_pkl=None,
):
    """
    Compute distance from each farm in cadastro to the nearest slaughterhouse
    and derive a continuous proximity risk score.

    Risk is 1 at distance = 0, decreases linearly to 0 at 5 km, and is exactly
    0 for any distance > 5 km.

    Parameters
    ----------
    cadastro_pkl : str or Path
        Path to the cadastro pickle (farm-level data with LATITUDE_DECIMAL /
        LONGITUDE_DECIMAL columns).
    movements_pkl : str or Path
        Path to the movements pickle (GTA records).
    out_pkl : str or Path or None
        If provided, save the updated cadastro dataframe to this path.

    Returns
    -------
    pd.DataFrame
        A copy of the cadastro dataframe with two new columns:
        - slaughterhouse_distance_km
        - slaughterhouse_risk
    """
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    with open(str(cadastro_pkl), "rb") as f:
        farms = pickle.load(f)

    with open(str(movements_pkl), "rb") as f:
        movements = pickle.load(f)

    # ------------------------------------------------------------------
    # 2. Extract slaughterhouse destinations
    # ------------------------------------------------------------------
    slaughter = movements[
        movements["FINALIDADE"].isin(["Abate", "Abate San."])
    ].copy()

    # Keep only records where the destination code looks like a slaughterhouse
    # (short CNPJ/code with < 6 digits, as specified)
    slaughter = slaughter[
        slaughter["COD_PROPRIEDADE_DESTINO"].astype(str).str.len() < 6
    ]

    slaughter_coords = (
        slaughter[["COD_PROPRIEDADE_DESTINO", "LATITUDE_DEC_DESTINO", "LONGITUDE_DEC_DESTINO"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    slaughter_coords["LATITUDE_DEC_DESTINO"] = pd.to_numeric(
        slaughter_coords["LATITUDE_DEC_DESTINO"], errors="coerce"
    )
    slaughter_coords["LONGITUDE_DEC_DESTINO"] = pd.to_numeric(
        slaughter_coords["LONGITUDE_DEC_DESTINO"], errors="coerce"
    )
    slaughter_coords = slaughter_coords.dropna(
        subset=["LATITUDE_DEC_DESTINO", "LONGITUDE_DEC_DESTINO"]
    )
    slaughter_coords = slaughter_coords[
        np.isfinite(slaughter_coords["LATITUDE_DEC_DESTINO"])
        & np.isfinite(slaughter_coords["LONGITUDE_DEC_DESTINO"])
    ].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Build BallTree on slaughterhouse coordinates (haversine)
    # ------------------------------------------------------------------
    # BallTree expects (lat, lon) in radians
    slaughter_rad = np.deg2rad(
        slaughter_coords[["LATITUDE_DEC_DESTINO", "LONGITUDE_DEC_DESTINO"]].values
    )
    tree = BallTree(slaughter_rad, metric="haversine")

    # ------------------------------------------------------------------
    # 4. Query nearest slaughterhouse for every farm
    # ------------------------------------------------------------------
    farms = farms.copy()
    farms["LATITUDE_DECIMAL"] = pd.to_numeric(farms["LATITUDE_DECIMAL"], errors="coerce")
    farms["LONGITUDE_DECIMAL"] = pd.to_numeric(farms["LONGITUDE_DECIMAL"], errors="coerce")

    valid_mask = (
        farms["LATITUDE_DECIMAL"].notna()
        & farms["LONGITUDE_DECIMAL"].notna()
        & np.isfinite(farms["LATITUDE_DECIMAL"])
        & np.isfinite(farms["LONGITUDE_DECIMAL"])
    )

    farms["slaughterhouse_distance_km"] = np.nan

    farm_rad = np.deg2rad(
        farms.loc[valid_mask, ["LATITUDE_DECIMAL", "LONGITUDE_DECIMAL"]].values
    )
    dist_rad, _ = tree.query(farm_rad, k=1)
    dist_km = dist_rad[:, 0] * EARTH_RADIUS_KM

    farms.loc[valid_mask, "slaughterhouse_distance_km"] = dist_km

    # ------------------------------------------------------------------
    # 5. Derive continuous proximity risk score
    #    risk = max(0, 1 - d / cutoff)  → linear decay, 0 beyond cutoff
    # ------------------------------------------------------------------
    d = farms["slaughterhouse_distance_km"]
    farms["slaughterhouse_risk"] = np.where(
        d.isna(),
        np.nan,
        np.clip(1.0 - d / SLAUGHTER_CUTOFF_KM, 0.0, 1.0),
    )

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    if out_pkl:
        farms.to_pickle(str(out_pkl))

    return farms


slaughterhouse_df = slaughterhouse_proximity_risk(
    cadastro_pkl=CADASTRO_PKL,
    movements_pkl=MOVEMENTS_PKL,
    out_pkl=SLAUGHTERHOUSE_PKL,
)


# ============================================
# SUMMARY OF OUTPUTS
# ============================================
# Four separate feature sets saved:
#
# 1. NETWORK ANALYSIS
#    → intermediate/network_farms_rank.pkl
#    Columns: farm_id, in_degree, out_degree, total_degree (weighted & unweighted), network_risk_score
#
# 2. FARM DENSITY PRESSURE
#    → intermediate/farm_density_pressure.pkl
#    Columns: n_neighbors_5km, pressure_rank, n_neighbors_5km_normalized
#
# 3. ANIMAL PRESSURE
#    → intermediate/animal_pressure_within_radius.pkl
#    Columns: birds_within_5km, birds_pressure_normalized (by species)
#    Also individual species files saved to intermediate/
#
# 4. DISTANCE TO ROADS
#    → intermediate/distance_to_roads.pkl
#    Columns: dist_road_1_m, dist_road_2_m, dist_road_3_m, dist_3_roads_sum_m
#
# 5. SLAUGHTERHOUSE PROXIMITY RISK
#    → intermediate/slaughterhouse_proximity_risk.pkl
#    Columns: slaughterhouse_distance_km, slaughterhouse_risk
#    Risk score: linear decay [0, 1] within 5 km, 0 beyond
# ============================================

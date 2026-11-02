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
    CADASTRO_PKL, MOVEMENTS_PKL, ROADS_FULL_SHP, INTERMEDIATE,
    NETWORK_PKL, FARM_DENSITY_PKL, ANIMAL_PRESSURE_PKL,
    DISTANCE_ROADS_PKL, SLAUGHTERHOUSE_PKL,
)

# Optional subset of components to (re)compute, e.g.
#   python3 3_get_spatial_net_features.py farm_density animal_pressure
# With no (recognised) arguments every component runs, as before.
STEP3_COMPONENTS = {"network", "farm_density", "animal_pressure", "roads", "slaughterhouse"}
RUN_COMPONENTS = (set(sys.argv[1:]) & STEP3_COMPONENTS) or STEP3_COMPONENTS
print(f"Step 3 components to run: {sorted(RUN_COMPONENTS)}")

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
            "species_group",
            "animal_total",
        ]
    ].dropna(subset=["COD_PROPRIEDADE_ORIGEM", "COD_PROPRIEDADE_DESTINO", "animal_total"])

    def normalize_farm_id(series):
        s = series.astype(str).str.strip()
        s = s.str.replace(r"\\.0$", "", regex=True)
        s = s.str.replace(r"\\D", "", regex=True)
        s = s.replace("", np.nan)
        return s

    df["COD_PROPRIEDADE_ORIGEM"] = normalize_farm_id(df["COD_PROPRIEDADE_ORIGEM"])
    df["COD_PROPRIEDADE_DESTINO"] = normalize_farm_id(df["COD_PROPRIEDADE_DESTINO"])
    df["animal_total"] = pd.to_numeric(df["animal_total"], errors="coerce")
    # parse the emission date (Brazilian day-first format, matching 3_1_road_animal_preassure.py)
    df["DATA_EMISSAO"] = pd.to_datetime(df["DATA_EMISSAO"], dayfirst=True, errors="coerce")

    df = df.dropna(subset=["COD_PROPRIEDADE_ORIGEM", "COD_PROPRIEDADE_DESTINO", "animal_total"])

    def compute_loyalty(mov):
        """Temporal trading-partner loyalty per farm.

        For each farm and species we tile the farm's movement history into
        non-overlapping windows (365 days for cattle/buffalo and small
        ruminants, 183 days for swine), build the set of unique trading
        partners in each window, and compute the Jaccard similarity between
        every consecutive window pair. A farm's species loyalty is
        ``1 - mean(Jaccard)`` (so values close to 1 mean the farm keeps
        changing partners = low loyalty = higher risk), and its overall
        loyalty is the mean across the species it trades. Farms with no
        eligible consecutive window pair get NaN (imputed by the mean upstream).

        Returns a DataFrame with columns [farm_id, loyalty].
        """
        WINDOW_DAYS = {
            "bovine_bubalinos": 365,
            "ovinos_caprinos": 365,
            "suinos": 183,
        }

        m = mov.dropna(subset=["DATA_EMISSAO", "species_group"]).copy()
        # long farm -> partner table: each movement contributes two rows so a
        # partner counts whether the focal farm is the origin or the destination
        origin_side = m.rename(
            columns={"COD_PROPRIEDADE_ORIGEM": "farm_id",
                     "COD_PROPRIEDADE_DESTINO": "partner"}
        )[["farm_id", "partner", "species_group", "DATA_EMISSAO"]]
        dest_side = m.rename(
            columns={"COD_PROPRIEDADE_DESTINO": "farm_id",
                     "COD_PROPRIEDADE_ORIGEM": "partner"}
        )[["farm_id", "partner", "species_group", "DATA_EMISSAO"]]
        long = pd.concat([origin_side, dest_side], ignore_index=True)
        long = long[long["species_group"].isin(WINDOW_DAYS)]

        records = []
        for (farm_id, species), grp in long.groupby(["farm_id", "species_group"]):
            window_days = WINDOW_DAYS[species]
            t0 = grp["DATA_EMISSAO"].min()
            win_idx = ((grp["DATA_EMISSAO"] - t0).dt.days // window_days)
            # unique partner set per non-overlapping window
            partners_by_window = (
                grp.assign(_win=win_idx)
                   .groupby("_win")["partner"]
                   .apply(lambda s: set(s.dropna()))
            )
            keys = sorted(partners_by_window.index)
            jaccards = []
            for prev_k, cur_k in zip(keys[:-1], keys[1:]):
                a = partners_by_window[prev_k]
                b = partners_by_window[cur_k]
                if not a or not b:
                    continue
                union = a | b
                if not union:
                    continue
                jaccards.append(len(a & b) / len(union))
            if jaccards:
                records.append(
                    (farm_id, species, 1.0 - float(np.mean(jaccards)))
                )

        if not records:
            return pd.DataFrame({"farm_id": [], "loyalty": []})

        sp_loyalty = pd.DataFrame(
            records, columns=["farm_id", "species_group", "species_loyalty"]
        )
        farm_loyalty = (
            sp_loyalty.groupby("farm_id", as_index=False)["species_loyalty"]
                      .mean()
                      .rename(columns={"species_loyalty": "loyalty"})
        )
        return farm_loyalty

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

    # ------------------------------------------------------------------
    # Temporal loyalty: fold the trading-partner consistency metric into
    # the network component with 70% network / 30% loyalty weighting.
    # ------------------------------------------------------------------
    loyalty_df = compute_loyalty(df)
    degree_df = degree_df.merge(loyalty_df, on="farm_id", how="left")
    # farms with no eligible window pair get the mean of the distribution
    degree_df["loyalty"] = degree_df["loyalty"].fillna(degree_df["loyalty"].mean())
    # keep the pre-loyalty score for auditing, then combine
    degree_df["network_risk_base"] = degree_df["network_risk_score"]
    degree_df["network_risk_score"] = (
        0.70 * degree_df["network_risk_base"]
        + 0.30 * degree_df["loyalty"]
    ).clip(0, 1)

    if out_pkl:
        degree_df.to_pickle(str(out_pkl))

    return degree_df

if "network" in RUN_COMPONENTS:
    network = build_movement_network(
        MOVEMENTS_PKL,
        out_pkl=NETWORK_PKL,
    )


#Farm density within radius
def farm_density_pressure(gdf, radius_m=10000, id_col="COD_PROPRIEDADE_RURAL", out_pkl=None):
    # drop rows without geometry or with invalid coords
    gdf = gdf.dropna(subset=["geometry"]).copy()
    # cadastro is long-format (one row per farm-species); count FARMS, not
    # rows, so keep one point per farm (first declared coordinate) before
    # building the tree — otherwise multi-species farms count their own
    # extra rows, and their neighbours' extra rows, as neighbouring farms.
    gdf = gdf.drop_duplicates(subset=[id_col], keep="first")
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

if "farm_density" in RUN_COMPONENTS:
    farm_density_pressuredf = farm_density_pressure(
        gdf=gdf,
        radius_m=10000,
        out_pkl=FARM_DENSITY_PKL,
    )

# ============================================
# ANIMAL PRESSURE (host density within 10 km, geodesic, species-weighted)
# ============================================
EARTH_RADIUS_KM = 6371.0

# Species-specific FMDV shedding weight, relative to cattle = 1.
# Pigs excrete airborne FMDV at ~50-100x the rate of cattle/sheep for most
# strains (up to ~1000x for high-shedding strains like C Noville).
# Sources: Donaldson, Herniman, Parker & Sellers (1970, J Hyg Cambridge);
# Alexandersen & Donaldson (2002, Epidemiology and Infection); synthesized
# in Sørensen et al. (2003, DMI Scientific Report SR03-17).
# w=50 is the conservative end of the typical-strain range.
# Bubalinos (buffalo) is an ASSUMPTION: no buffalo-specific FMDV shedding
# data was found; treated as cattle-equivalent (both large ruminant bovids).
SPECIES_WEIGHTS = {
    "Suíno": 50,
    "Bovinos": 1,
    "Ovino": 1,
    "Caprinos": 1,
    "Bubalinos": 1,
}


def animal_pressure_weighted_10km(
    gdf,
    radius_km=10,
    species_col="ESPECIE_ANIMAL",
    capacity_col="QTD_DECLARADA",
    id_col="COD_PROPRIEDADE_RURAL",
    lat_col="LATITUDE_DECIMAL",
    lon_col="LONGITUDE_DECIMAL",
    out_pkl=None,
):
    """
    Farm-level animal pressure: for each farm i, the species-weighted sum of
    FMD-susceptible headcounts at all OTHER farms j within radius_km true
    geodesic (haversine) distance, excluding farm i's own rows:

        A_i = sum_{j != i, d(i,j) <= radius_km} w(species_j) * Q_j

    A farm may have more than one row (one per FMD-susceptible species it
    declared); self-exclusion is applied at the farm level — summing out ALL
    of that farm's own rows — not the row level. Most farms' rows share an
    identical coordinate, but a small number show coordinate drift between
    declarations (re-geocoding between visits); for those, querying from only
    one representative row would silently miss the drifted row's own
    neighborhood and could subtract more "own weight" than was ever added,
    producing a negative pressure value. See the fast/slow path split below.
    """
    df = gdf[[id_col, species_col, capacity_col, lat_col, lon_col]].dropna(
        subset=[species_col, capacity_col, lat_col, lon_col]
    ).copy()
    df[capacity_col] = pd.to_numeric(df[capacity_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[capacity_col, lat_col, lon_col]).reset_index(drop=True)

    unmapped = set(df[species_col].unique()) - set(SPECIES_WEIGHTS)
    if unmapped:
        raise ValueError(f"Unmapped species in SPECIES_WEIGHTS: {sorted(unmapped)}")
    df["row_weight"] = df[species_col].map(SPECIES_WEIGHTS) * df[capacity_col]

    coords_rad = np.deg2rad(df[[lat_col, lon_col]].values)
    tree = BallTree(coords_rad, metric="haversine")
    radius_rad = radius_km / EARTH_RADIUS_KM

    row_weight = df["row_weight"].values
    capacity = df[capacity_col].values
    is_swine = (df[species_col] == "Suíno").values
    swine_weight = SPECIES_WEIGHTS["Suíno"]

    farm_row_positions = df.groupby(id_col).indices
    unique_farm_ids = np.array(list(farm_row_positions.keys()))

    # Almost every farm's own rows share one coordinate, so one query per
    # farm (using a representative row) is enough and is fully vectorized.
    # A small minority show coordinate drift between declarations
    # (re-geocoding between visits) — those need the slower per-row
    # neighbor union below. Flagging them up front keeps the fast path fast
    # instead of paying union/dedup cost for every farm.
    coord_nunique = df.groupby(id_col)[[lat_col, lon_col]].nunique()
    drifted_farm_ids = set(
        coord_nunique.index[(coord_nunique[lat_col] > 1) | (coord_nunique[lon_col] > 1)]
    )
    simple_farm_ids = [f for f in unique_farm_ids if f not in drifted_farm_ids]

    def _farm_pressure(neigh, own_pos):
        total_weighted = row_weight[neigh].sum()
        own_weighted = row_weight[own_pos].sum()
        pressure = total_weighted - own_weighted

        swine_neighbor_qty_raw = (
            capacity[neigh][is_swine[neigh]].sum()
            - capacity[own_pos][is_swine[own_pos]].sum()
        )
        nonswine_weighted_neighbor_qty = pressure - swine_weight * swine_neighbor_qty_raw
        return pressure, swine_neighbor_qty_raw, nonswine_weighted_neighbor_qty

    records = []

    if simple_farm_ids:
        first_pos = np.array([farm_row_positions[f][0] for f in simple_farm_ids])
        neighbor_idx_simple = tree.query_radius(coords_rad[first_pos], r=radius_rad)
        for farm_id, neigh in zip(simple_farm_ids, neighbor_idx_simple):
            own_pos = farm_row_positions[farm_id]
            records.append((farm_id, *_farm_pressure(neigh, own_pos)))

    if drifted_farm_ids:
        drifted_positions = np.unique(
            np.concatenate([farm_row_positions[f] for f in drifted_farm_ids])
        )
        neighbor_idx_drifted = tree.query_radius(coords_rad[drifted_positions], r=radius_rad)
        pos_to_neighbors = dict(zip(drifted_positions, neighbor_idx_drifted))
        for farm_id in drifted_farm_ids:
            own_pos = np.asarray(farm_row_positions[farm_id])
            neigh = np.unique(np.concatenate([pos_to_neighbors[p] for p in own_pos]))
            records.append((farm_id, *_farm_pressure(neigh, own_pos)))

    out = pd.DataFrame(
        records,
        columns=[
            id_col,
            "animal_pressure_weighted_10km",
            "swine_neighbor_qty_raw",
            "nonswine_weighted_neighbor_qty",
        ],
    )
    x = out["animal_pressure_weighted_10km"].values.astype(float)
    out["animal_pressure_weighted_10km_norm"] = (
        (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.0
    )

    if out_pkl:
        out.to_pickle(str(out_pkl))

    return out


if __name__ == "__main__" and "animal_pressure" in RUN_COMPONENTS:
    animal_pressure_within_radius_df = animal_pressure_weighted_10km(
        gdf=gdf,
        radius_km=10,
        species_col="ESPECIE_ANIMAL",
        capacity_col="QTD_DECLARADA",
        id_col="COD_PROPRIEDADE_RURAL",
        lat_col="LATITUDE_DECIMAL",
        lon_col="LONGITUDE_DECIMAL",
        out_pkl=ANIMAL_PRESSURE_PKL,
    )
    print(
        f"  Animal pressure (species-weighted, 10 km geodesic): "
        f"{animal_pressure_within_radius_df['COD_PROPRIEDADE_RURAL'].nunique():,} farms"
    )



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

if "roads" in RUN_COMPONENTS:
    import multiprocessing as mp
    # Processes only where fork is the start method (Linux): under spawn
    # (macOS default) each worker would re-import this module and re-run
    # every module-level component, so keep the 4-thread setup there.
    _fork = mp.get_start_method() == "fork"
    _road_workers = min(48, os.cpu_count() or 4) if _fork else 4
    # cadastro is long-format; distances depend only on the farm point, so
    # compute once per farm (first declared coordinate, as in step 7).
    _farms_once = gdf.drop_duplicates(subset=["COD_PROPRIEDADE_RURAL"], keep="first")
    distance_to_roadsdf = distance_to_roads_efficient(
        gdf_farms=_farms_once,
        roads_shp=ROADS_FULL_SHP,
        epsg_m=31982,
        out_pkl=DISTANCE_ROADS_PKL,
        parallel=True,
        max_workers=_road_workers,
        # one chunk per worker so the roads layer is pickled once per worker
        chunk_size=-(-len(_farms_once) // _road_workers) if _fork else 2000,
        parallel_backend="process" if _fork else "thread",  # thread is safe in interactive
    )

# ============================================
# SLAUGHTERHOUSE PROXIMITY RISK
# ============================================
# EARTH_RADIUS_KM is defined above, in the animal-pressure section.
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


if "slaughterhouse" in RUN_COMPONENTS:
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
#    Columns: farm_id, in_degree, out_degree, total_degree (weighted & unweighted),
#             loyalty, network_risk_base, network_risk_score
#    network_risk_score = 0.70 * network_risk_base + 0.30 * temporal loyalty
#
# 2. FARM DENSITY PRESSURE
#    → intermediate/farm_density_pressure.pkl
#    Columns: n_neighbors_5km, pressure_rank, n_neighbors_5km_normalized
#
# 3. ANIMAL PRESSURE (species-weighted, 10 km geodesic)
#    → intermediate/animal_pressure_within_radius.pkl
#    Columns: animal_pressure_weighted_10km, animal_pressure_weighted_10km_norm,
#             swine_neighbor_qty_raw, nonswine_weighted_neighbor_qty
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

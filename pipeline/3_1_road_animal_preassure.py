# =============================================================================
# ROAD ANIMAL PRESSURE PIPELINE
# Converts livestock movements into road-based animal pressure and farm exposure
# using OSM road network routing (graph built from local roads shapefile).
# =============================================================================

# =============================================================================
# STEP 1: LOAD LIBRARIES
# =============================================================================
import gc
import multiprocessing as mp
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.ops import linemerge
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import (
    MOVEMENTS_PKL, CADASTRO_PKL, ROADS_FULL_SHP,
    GRAPH_RAW_CACHE_FULL, GRAPH_SIMP_CACHE_FULL, ROUTE_CACHE_FULL,
    TMP_MONTHLY, TMP_PRESSURE_FULL, ROAD_OUT,
)

# =============================================================================
# PATHS & SETTINGS
# =============================================================================
ROADS_SHP        = ROADS_FULL_SHP
GRAPH_RAW_CACHE  = GRAPH_RAW_CACHE_FULL
GRAPH_SIMP_CACHE = GRAPH_SIMP_CACHE_FULL
ROUTE_CACHE      = ROUTE_CACHE_FULL
TMP_PRESSURE     = TMP_PRESSURE_FULL

CRS_GEO      = "EPSG:4326"
CRS_METRIC   = "EPSG:32722"   # UTM Zone 22S — appropriate for RS, Brazil
BUFFER_M     = 10_000          # 10 km buffer in metres
CHUNK_FARMS  = 500             # farms per buffer-intersection batch
N_WORKERS    = max(1, mp.cpu_count() - 1)   # leave one core free


# =============================================================================
# MODULE-LEVEL ROUTING STATE (populated before Pool creation, inherited by
# fork workers — avoids re-serialising the graph to each worker process).
#
# Routing is done as one scipy sparse Dijkstra run per distinct ORIGIN node
# rather than one networkx shortest_path call per (origin, destination) PAIR.
# With route pairs sharing a much smaller set of distinct origins
# (farms/slaughterhouses repeat heavily as movement endpoints), the per-pair
# approach means re-solving overlapping shortest-path trees from scratch
# hundreds of thousands of times; per-origin single-source Dijkstra solves
# each origin's tree once and reads off every needed destination from it.
# =============================================================================
_CSR_ROUTING = None  # sparse adjacency matrix, set in main body before Pool
_NODE_IDS    = None  # node_ids list; _NODE_IDS[i] is the row/col i node id


def _build_csr_graph(G: nx.Graph, node_index: dict, n_nodes: int) -> csr_matrix:
    """Undirected sparse adjacency matrix for scipy's Dijkstra, weighted by
    edge length (metres). Built once from the already-cached simplified graph."""
    rows, cols, weights = [], [], []
    for u, v, data in G.edges(data=True):
        i, j = node_index[u], node_index[v]
        w = float(data.get("length", 0.0))
        rows.append(i); cols.append(j); weights.append(w)
        rows.append(j); cols.append(i); weights.append(w)
    return csr_matrix((weights, (rows, cols)), shape=(n_nodes, n_nodes))


def _mp_route_from_origin(job):
    """Worker: single-source Dijkstra from one origin node to every target
    node it needs a route to, returning (origin_idx, {target_idx: path})."""
    origin_idx, target_idxs = job
    _, predecessors = sp_dijkstra(
        csgraph=_CSR_ROUTING, directed=False, indices=origin_idx,
        return_predecessors=True,
    )
    results = {}
    for t in target_idxs:
        path_idx = [t]
        cur = t
        unreachable = False
        while cur != origin_idx:
            nxt = predecessors[cur]
            if nxt == -9999:
                unreachable = True
                break
            path_idx.append(nxt)
            cur = nxt
        if unreachable:
            results[t] = None
        else:
            path_idx.reverse()
            results[t] = [_NODE_IDS[i] for i in path_idx]
    return origin_idx, results


# =============================================================================
# GRAPH SIMPLIFICATION HELPER
# Contracts degree-2 nodes into single edges, preserving geometry.
# Reduces a ~1 M node coordinate graph to ~100 k intersection nodes,
# making Dijkstra 10–50x faster.
# =============================================================================
def _simplify_graph(G: nx.Graph) -> nx.Graph:
    keep = frozenset(n for n in G.nodes() if G.degree(n) != 2)
    if not keep:
        return G

    simplified = nx.Graph()
    seen: set = set()

    for src in keep:
        for nbr in list(G.neighbors(src)):
            canon = (src, nbr) if src <= nbr else (nbr, src)
            if canon in seen:
                continue
            seen.add(canon)

            segs = [G[src][nbr].get("geometry", LineString([src, nbr]))]
            total_len = float(G[src][nbr].get("length", 0.0))
            prev, cur = src, nbr

            while cur not in keep:
                nbs = [n for n in G.neighbors(cur) if n != prev]
                if not nbs:
                    break
                nxt = nbs[0]
                nxt_canon = (cur, nxt) if cur <= nxt else (nxt, cur)
                seen.add(nxt_canon)
                ed = G[cur][nxt]
                segs.append(ed.get("geometry", LineString([cur, nxt])))
                total_len += float(ed.get("length", 0.0))
                prev, cur = cur, nxt

            dst = cur
            if src == dst:
                continue   # skip self-loops

            try:
                merged = linemerge(segs)
                if merged.geom_type == "MultiLineString":
                    merged = max(merged.geoms, key=lambda g: g.length)
            except Exception:
                merged = LineString([src, dst])

            simplified.add_edge(src, dst, length=total_len, geometry=merged)

    return simplified


def _edge_geom(G: nx.Graph, u, v) -> LineString:
    data = G.get_edge_data(u, v) or G.get_edge_data(v, u) or {}
    return data.get("geometry", LineString([u, v]))


# =============================================================================
# STEP 2: LOAD DATA
# =============================================================================
print("\n" + "="*60)
print("STEP 2: Loading data...")

with open(str(MOVEMENTS_PKL), "rb") as f:
    mov = pickle.load(f)
print(f"  Movements       : {len(mov):,} rows")

with open(str(CADASTRO_PKL), "rb") as f:
    farms = pickle.load(f)
print(f"  Farms (cadastro): {len(farms):,} rows")


# =============================================================================
# STEP 3: DATA CLEANING
# =============================================================================
print("\nSTEP 3: Cleaning data...")

mov["DATA_EMISSAO"] = pd.to_datetime(mov["DATA_EMISSAO"], dayfirst=True, errors="coerce")

if "year_month" not in mov.columns:
    mov["year_month"] = mov["DATA_EMISSAO"].dt.to_period("M").astype(str)

def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

mov["COD_PROPRIEDADE_ORIGEM"]  = _norm(mov["COD_PROPRIEDADE_ORIGEM"])
mov["COD_PROPRIEDADE_DESTINO"] = _norm(mov["COD_PROPRIEDADE_DESTINO"])
farms["COD_PROPRIEDADE_RURAL"] = _norm(farms["COD_PROPRIEDADE_RURAL"])

n_before = len(mov)
mov = mov.dropna(subset=["COD_PROPRIEDADE_ORIGEM", "COD_PROPRIEDADE_DESTINO"])
mov = mov[
    (mov["COD_PROPRIEDADE_ORIGEM"].str.len() > 0) &
    (mov["COD_PROPRIEDADE_DESTINO"].str.len() > 0)
]
print(f"  Dropped {n_before - len(mov):,} rows with missing origin/destination IDs.")

coord_lkp = (
    farms[["COD_PROPRIEDADE_RURAL", "LATITUDE_DECIMAL", "LONGITUDE_DECIMAL"]]
    .dropna()
    .drop_duplicates("COD_PROPRIEDADE_RURAL")
    .set_index("COD_PROPRIEDADE_RURAL")
)

mov = mov.merge(
    coord_lkp.rename(columns={"LATITUDE_DECIMAL": "lat_o", "LONGITUDE_DECIMAL": "lon_o"}),
    left_on="COD_PROPRIEDADE_ORIGEM",
    right_index=True,
    how="left",
)

mov["lat_d"] = pd.to_numeric(mov["LATITUDE_DEC_DESTINO"],  errors="coerce")
mov["lon_d"] = pd.to_numeric(mov["LONGITUDE_DEC_DESTINO"], errors="coerce")

n_before = len(mov)
mov = mov.dropna(subset=["lat_o", "lon_o", "lat_d", "lon_d", "animal_total"])
print(f"  Dropped {n_before - len(mov):,} rows with unresolved coordinates.")

_KEEP = [
    "DATA_EMISSAO", "FINALIDADE",
    "LATITUDE_DEC_DESTINO", "LONGITUDE_DEC_DESTINO",
    "COD_MUN_IBGE_ORIGEM", "COD_PROPRIEDADE_ORIGEM", "COD_PROPRIEDADE_DESTINO",
    "COD_MUN_IBGE_DESTINO", "species_group", "animal_total", "year_month",
    "lat_o", "lon_o", "lat_d", "lon_d",
]
mov = mov[[c for c in _KEEP if c in mov.columns]]
print(f"  Clean movements: {len(mov):,} rows, {mov['year_month'].nunique()} months")


# =============================================================================
# STEP 4: CREATE UNDIRECTED ROUTE ID
# =============================================================================
print("\nSTEP 4: Creating undirected route IDs...")
_o = mov["COD_PROPRIEDADE_ORIGEM"]
_d = mov["COD_PROPRIEDADE_DESTINO"]
mov["route_id"] = np.where(_o <= _d, _o + "_" + _d, _d + "_" + _o)
print(f"  {mov['route_id'].nunique():,} unique undirected routes")


# =============================================================================
# STEP 5: MONTHLY AGGREGATION (CHUNKED) + STEP 5.1 VALIDATION
# =============================================================================
print("\nSTEP 5: Monthly aggregation (chunked by year_month)...")
months = sorted(mov["year_month"].unique())
print(f"  Months to process: {len(months)}")

for ym in tqdm(months, desc="Monthly aggregation"):
    out_path = TMP_MONTHLY / f"{ym}.pkl"
    if out_path.exists():
        _agg = pd.read_pickle(out_path)
        print(f"  [{ym}] (cached) {_agg['route_id'].nunique():,} routes | "
              f"{int(_agg['animal_total'].sum()):,} animals")
        continue

    sub = mov[mov["year_month"] == ym]
    agg = (
        sub.groupby(["route_id", "species_group"])["animal_total"]
        .sum()
        .reset_index()
    )

    # STEP 5.1: FLOW CREATION CHECK — HARD CHECKPOINT
    _errors = []
    if agg.empty:
        _errors.append("no aggregated flows created (empty dataframe)")
    if not agg.empty and agg["species_group"].isnull().any():
        _errors.append("missing values found in 'species_group'")
    if not agg.empty and agg["animal_total"].isnull().any():
        _errors.append("missing values found in 'animal_total' after aggregation")
    if not agg.empty and agg["route_id"].nunique() == 0:
        _errors.append("no valid route_ids found")

    if _errors:
        print(f"\n[VALIDATION ERROR] Month {ym} failed the flow creation check:")
        for _e in _errors:
            print(f"  - {_e}")
        print("Stopping execution immediately. Fix input data before re-running.")
        raise SystemExit(1)

    print(f"  [{ym}] {agg['route_id'].nunique():,} routes | "
          f"{int(agg['animal_total'].sum()):,} total animals")
    agg.to_pickle(out_path)
    del sub, agg
    gc.collect()


# =============================================================================
# STEP 6: ROUTE COORDINATES LOOKUP TABLE
# =============================================================================
print("\nSTEP 6: Building route coordinate lookup from farms table...")
route_coords = (
    mov[["route_id", "lat_o", "lon_o", "lat_d", "lon_d"]]
    .drop_duplicates("route_id")
    .reset_index(drop=True)
)
print(f"  {len(route_coords):,} unique routes with coordinates")

del mov
gc.collect()


# =============================================================================
# STEP 7: LOAD ROAD NETWORK (OSM roads shapefile → simplified networkx graph)
# Builds from the full road network (raw_data/roads/roads_rs_full.shp, every
# drivable OSM highway class — built once by pipeline/3_0_download_full_roads.py),
# then contracts degree-2 nodes so only intersections and dead-ends remain.
# Reduces a multi-million node graph to a much smaller set of nodes, making
# Dijkstra 10–50x faster.
# =============================================================================
print("\nSTEP 7: Loading OSM road network...")

if GRAPH_SIMP_CACHE.exists():
    print("  Loading simplified graph from cache...")
    with open(str(GRAPH_SIMP_CACHE), "rb") as f:
        G, node_ids, node_arr = pickle.load(f)
    print(f"  Simplified graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges")

else:
    # Build raw graph (or load previously cached raw graph)
    if GRAPH_RAW_CACHE.exists():
        print("  Loading raw graph from cache...")
        with open(str(GRAPH_RAW_CACHE), "rb") as f:
            G_raw, _, _ = pickle.load(f)
        print(f"  Raw graph: {len(G_raw.nodes):,} nodes, {len(G_raw.edges):,} edges")
    else:
        if not ROADS_SHP.exists():
            raise FileNotFoundError(
                f"Full road network shapefile not found: {ROADS_SHP}\n"
                "Run pipeline/3_0_download_full_roads.py first to build it "
                "(see that script's header for the required Python environment)."
            )
        roads = gpd.read_file(str(ROADS_SHP)).to_crs(CRS_METRIC)
        print(f"  Loaded {len(roads):,} road segments from {ROADS_SHP.name}")
        G_raw = nx.Graph()
        for _, row in tqdm(roads.iterrows(), total=len(roads), desc="Building raw graph"):
            if row.geometry is None or row.geometry.is_empty:
                continue
            coords = list(row.geometry.coords)
            for i in range(len(coords) - 1):
                u, v = coords[i], coords[i + 1]
                seg = LineString([u, v])
                G_raw.add_edge(u, v, length=seg.length, geometry=seg,
                               highway=str(row.get("highway", "")))
        largest_cc = max(nx.connected_components(G_raw), key=len)
        G_raw = G_raw.subgraph(largest_cc).copy()
        _node_ids_raw = list(G_raw.nodes())
        with open(str(GRAPH_RAW_CACHE), "wb") as f:
            pickle.dump((G_raw, _node_ids_raw, np.array(_node_ids_raw)), f)
        print(f"  Raw graph: {len(G_raw.nodes):,} nodes, {len(G_raw.edges):,} edges")

    # Simplify: contract all degree-2 nodes
    print("  Simplifying graph (contracting degree-2 nodes)...")
    G = _simplify_graph(G_raw)
    del G_raw
    gc.collect()

    node_ids = list(G.nodes())
    node_arr = np.array(node_ids)
    with open(str(GRAPH_SIMP_CACHE), "wb") as f:
        pickle.dump((G, node_ids, node_arr), f)
    print(f"  Simplified graph: {len(G.nodes):,} nodes, {len(G.edges):,} edges")

# KD-tree for fast nearest-node lookup
tree = cKDTree(node_arr)
print(f"  KD-tree ready ({len(node_ids):,} nodes) | {N_WORKERS} routing workers")


# =============================================================================
# STEP 8: SHORTEST PATH COMPUTATION (PER-ORIGIN + DISK CACHE + MULTIPROCESSING)
# One scipy sparse Dijkstra run per distinct origin node (see _mp_route_from_origin
# above for why), parallelised across workers via fork so each inherits the
# adjacency matrix without re-serialising it.
# =============================================================================
print("\nSTEP 8: Computing shortest paths...")

try:
    with open(str(ROUTE_CACHE), "rb") as f:
        route_cache = pickle.load(f)
    print(f"  Loaded {len(route_cache):,} cached paths from disk")
except (FileNotFoundError, EOFError, pickle.UnpicklingError):
    # Truncated cache (process killed mid-write) — start fresh
    route_cache = {}

# Project coordinates to CRS_METRIC for KD-tree
coords_o = gpd.GeoDataFrame(
    route_coords[["route_id"]],
    geometry=gpd.points_from_xy(route_coords["lon_o"], route_coords["lat_o"]),
    crs=CRS_GEO,
).to_crs(CRS_METRIC)
route_coords["x_o"] = coords_o.geometry.x
route_coords["y_o"] = coords_o.geometry.y

coords_d = gpd.GeoDataFrame(
    route_coords[["route_id"]],
    geometry=gpd.points_from_xy(route_coords["lon_d"], route_coords["lat_d"]),
    crs=CRS_GEO,
).to_crs(CRS_METRIC)
route_coords["x_d"] = coords_d.geometry.x
route_coords["y_d"] = coords_d.geometry.y
del coords_o, coords_d
gc.collect()

print(f"  Snapping {len(route_coords):,} route endpoints to nearest graph nodes...")
_, idx_o = tree.query(route_coords[["x_o", "y_o"]].values)
_, idx_d = tree.query(route_coords[["x_d", "y_d"]].values)

route_coords["o_node"] = [node_ids[i] for i in idx_o]
route_coords["d_node"] = [node_ids[i] for i in idx_d]
route_coords["pair"]   = [
    tuple(sorted((a, b)))
    for a, b in zip(route_coords["o_node"], route_coords["d_node"])
]

all_pairs   = route_coords["pair"].unique().tolist()
pairs_todo  = [p for p in all_pairs if p not in route_cache]
print(f"  {len(pairs_todo):,} new pairs to route | {len(route_cache):,} already cached")

if pairs_todo:
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    print("  Building sparse adjacency matrix for per-origin routing...")
    csr = _build_csr_graph(G, node_index, len(node_ids))

    # Group remaining pairs by origin node index: one Dijkstra run per
    # distinct origin covers every destination that origin needs at once.
    origin_targets = defaultdict(list)
    for a, b in pairs_todo:
        origin_targets[node_index[a]].append(node_index[b])
    jobs = list(origin_targets.items())
    print(f"  {len(jobs):,} distinct origins to route from "
          f"(vs {len(pairs_todo):,} pairs)")

    # Expose routing state to fork workers via module-level globals
    _CSR_ROUTING = csr
    _NODE_IDS    = node_ids

    computed   = 0
    failed     = 0
    since_save = 0
    SAVE_EVERY_ORIGINS = 2_000  # flush route_cache to disk periodically

    # "fork" avoids re-serialising the graph/CSR matrix to each worker
    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS) as pool:
        for origin_idx, results in tqdm(
            pool.imap_unordered(_mp_route_from_origin, jobs),
            total=len(jobs), desc="Routing (per-origin Dijkstra)",
        ):
            origin_node = node_ids[origin_idx]
            for t_idx, path in results.items():
                target_node = node_ids[t_idx]
                pair = tuple(sorted((origin_node, target_node)))
                route_cache[pair] = path
                if path:
                    computed += 1
                else:
                    failed += 1

            since_save += 1
            if since_save >= SAVE_EVERY_ORIGINS:
                with open(str(ROUTE_CACHE), "wb") as f:
                    pickle.dump(route_cache, f)
                since_save = 0

    with open(str(ROUTE_CACHE), "wb") as f:
        pickle.dump(route_cache, f)

    print(f"  Routing complete — {computed:,} ok, {failed:,} failed")

route_coords["path"] = route_coords["pair"].map(route_cache)
route_lookup = route_coords.set_index("route_id")[["pair", "path"]]

del pairs_todo, idx_o, idx_d, all_pairs
gc.collect()


# =============================================================================
# STEP 9 & 10: BUILD MONTHLY ROUTE GEODATA (GPKG) & SPATIAL MERGE
# =============================================================================
print("\nSTEP 9 & 10: Building monthly road pressure (GPKG)...")

for pkl_file in tqdm(sorted(TMP_MONTHLY.glob("*.pkl")), desc="Monthly road pressure"):
    ym       = pkl_file.stem
    out_gpkg = TMP_PRESSURE / f"{ym}.gpkg"
    if out_gpkg.exists():
        continue

    agg = pd.read_pickle(pkl_file)
    agg = agg.merge(route_lookup[["path"]], on="route_id", how="left")
    agg = agg.dropna(subset=["path"])

    if agg.empty:
        print(f"  [{ym}] No routable paths — skipping.")
        continue

    edge_records = []
    for _, row in agg.iterrows():
        path = row["path"]
        if not path or len(path) < 2:
            continue
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            geom = _edge_geom(G, u, v)
            edge_records.append({
                "edge_wkt":      geom.wkt,
                "species_group": row["species_group"],
                "animal_total":  row["animal_total"],
            })

    if not edge_records:
        print(f"  [{ym}] No valid edge sequences — skipping.")
        continue

    edge_df = pd.DataFrame(edge_records)
    edge_pressure = (
        edge_df.groupby(["edge_wkt", "species_group"])["animal_total"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    edge_pressure.columns.name = None

    month_gdf = gpd.GeoDataFrame(
        edge_pressure.drop(columns="edge_wkt"),
        geometry=gpd.GeoSeries.from_wkt(edge_pressure["edge_wkt"], crs=CRS_METRIC),
        crs=CRS_METRIC,
    )
    month_gdf.to_file(str(out_gpkg), driver="GPKG")
    print(f"  [{ym}] {len(month_gdf):,} road segments -> {out_gpkg.name}")

    del agg, edge_df, edge_pressure, month_gdf, edge_records
    gc.collect()


# =============================================================================
# STEP 11 & 12: AVERAGE ROAD PRESSURE ACROSS MONTHS & SAVE SHAPEFILE
# =============================================================================
print("\nSTEP 11 & 12: Computing average road pressure and saving...")

gpkg_files = sorted(TMP_PRESSURE.glob("*.gpkg"))
if not gpkg_files:
    raise FileNotFoundError(
        "No monthly pressure GPKG files found. "
        "Pipeline may have failed during routing or edge assignment."
    )

monthly_dfs = []
for gf in tqdm(gpkg_files, desc="Loading monthly GPKG files"):
    gdf = gpd.read_file(str(gf))
    gdf["wkt"] = gdf.geometry.to_wkt()
    monthly_dfs.append(gdf.drop(columns="geometry"))
    del gdf

all_flows = pd.concat(monthly_dfs, ignore_index=True)
del monthly_dfs
gc.collect()

species_cols = [c for c in all_flows.columns if c != "wkt"]

avg_pressure = (
    all_flows.groupby("wkt")[species_cols]
    .mean()
    .reset_index()
)
avg_pressure["total_pressure"] = avg_pressure[species_cols].sum(axis=1)

avg_pressure_gdf = gpd.GeoDataFrame(
    avg_pressure,
    geometry=gpd.GeoSeries.from_wkt(avg_pressure["wkt"], crs=CRS_METRIC),
).drop(columns="wkt")

road_out = ROAD_OUT / "road_animal_pressure.shp"
avg_pressure_gdf.to_file(str(road_out))
print(f"  Saved: {len(avg_pressure_gdf):,} road segments -> {road_out}")

del all_flows, avg_pressure
gc.collect()


# =============================================================================
# STEP 13: FARM GEODATAFRAME + 10 KM BUFFERS
# =============================================================================
print("\nSTEP 13: Building farm GeoDataFrame and generating 10 km buffers...")

farms_clean  = farms.dropna(subset=["LATITUDE_DECIMAL", "LONGITUDE_DECIMAL"]).copy()
farms_unique = farms_clean.drop_duplicates(subset=["COD_PROPRIEDADE_RURAL"])

farm_gdf = gpd.GeoDataFrame(
    farms_unique[["COD_PROPRIEDADE_RURAL"]].copy(),
    geometry=gpd.points_from_xy(
        farms_unique["LONGITUDE_DECIMAL"],
        farms_unique["LATITUDE_DECIMAL"],
    ),
    crs=CRS_GEO,
).to_crs(CRS_METRIC)

farm_gdf["buffer_geom"] = farm_gdf.geometry.buffer(BUFFER_M)
print(f"  {len(farm_gdf):,} farms with valid coordinates")

del farms_clean, farms_unique
gc.collect()


# =============================================================================
# STEP 14: BUFFER INTERSECTION (CHUNKED) WITH SPATIAL INDEX
# =============================================================================
print("\nSTEP 14: Intersecting 10 km farm buffers with road pressure (chunked)...")

road_pressure  = avg_pressure_gdf.copy()
_pressure_cols = [c for c in road_pressure.columns if c != "geometry"]

farm_results = []
farm_chunks  = [
    farm_gdf.iloc[i: i + CHUNK_FARMS]
    for i in range(0, len(farm_gdf), CHUNK_FARMS)
]

for chunk in tqdm(farm_chunks, desc="Farm buffer chunks"):
    buf_gdf = gpd.GeoDataFrame(
        chunk[["COD_PROPRIEDADE_RURAL"]].copy(),
        geometry=chunk["buffer_geom"].values,
        crs=CRS_METRIC,
    )
    joined = gpd.sjoin(buf_gdf, road_pressure, how="inner", predicate="intersects")
    if joined.empty:
        continue

    _agg_cols = [c for c in _pressure_cols if c in joined.columns]
    farm_results.append(
        joined.groupby("COD_PROPRIEDADE_RURAL")[_agg_cols].mean().reset_index()
    )
    del buf_gdf, joined
    gc.collect()


# =============================================================================
# STEP 15: SAVE FARM OUTPUT
# =============================================================================
print("\nSTEP 15: Saving farm exposure output...")

if farm_results:
    farm_exposure     = pd.concat(farm_results, ignore_index=True)
    farm_exposure_gdf = (
        farm_gdf[["COD_PROPRIEDADE_RURAL", "geometry"]]
        .merge(farm_exposure, on="COD_PROPRIEDADE_RURAL", how="left")
    )
    _num = farm_exposure_gdf.select_dtypes("number").columns
    farm_exposure_gdf[_num] = farm_exposure_gdf[_num].fillna(0)
    farm_out = ROAD_OUT / "farm_road_pressure_10km.shp"
    farm_exposure_gdf.to_file(str(farm_out))
    print(f"  Saved: {len(farm_exposure_gdf):,} farms -> {farm_out}")
else:
    print("  [WARNING] No farm-buffer intersections found — farm output not saved.")


# =============================================================================
# STEP 16 & 17: MEMORY CLEANUP & FINAL LOG
# =============================================================================
gc.collect()

print("\n" + "="*60)
print("STEP 17 — Pipeline summary")
print(f"  Road pressure  -> {ROAD_OUT / 'road_animal_pressure.shp'}")
print(f"  Farm exposure  -> {ROAD_OUT / 'farm_road_pressure_10km.shp'}")
print(f"  Months processed : {len(sorted(TMP_MONTHLY.glob('*.pkl')))}")
print(f"  Pressure months  : {len(sorted(TMP_PRESSURE.glob('*.gpkg')))}")
print("DONE: Pipeline finished successfully.")

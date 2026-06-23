#!/usr/bin/env python3
"""
Compute distance from farms to the international border, and (optionally)
separate distances to river vs dry border segments.

Default inputs are aligned with this repo's data layout:
- Farms: intermediate/cadastro.pkl (LATITUDE_DECIMAL, LONGITUDE_DECIMAL)
- RS boundary: raw_data/rs_boundary_ibge/RS_UF_2024.shp (for clipping)

If you don't provide border/rivers shapefiles, you can use --auto-download
to fetch Natural Earth datasets.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import (
    CADASTRO_PKL, RS_BOUNDARY_SHP,
    BORDER_DIR, HYDRO_DIR, BORDER_PKL,
)

NE_BORDER_URL = (
    "https://naturalearth.s3.amazonaws.com/10m_cultural/"
    "ne_10m_admin_0_boundary_lines_land.zip"
)
NE_RIVERS_URL = (
    "https://naturalearth.s3.amazonaws.com/10m_physical/"
    "ne_10m_rivers_lake_centerlines.zip"
)


def _expand(path: str | Path) -> Path:
    return Path(str(path)).expanduser().resolve()


def _download_and_extract(url: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / Path(url).name
    if not zip_path.exists():
        print(f"Downloading: {url}")
        urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    # Return first .shp in extracted folder
    shp_files = list(dest_dir.glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp found after extracting {zip_path}")
    return shp_files[0]

def _union_all(geoms: gpd.GeoSeries):
    if hasattr(geoms, "union_all"):
        return geoms.union_all()
    return geoms.unary_union



def _load_border_lines(
    border_shp: Path,
    country: str = "Brazil",
    neighbors: Iterable[str] = ("Uruguay", "Argentina"),
) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(border_shp)

    # Attempt to filter by admin columns if present
    cols = {c.lower(): c for c in gdf.columns}
    left_col = cols.get("adm0_left")
    right_col = cols.get("adm0_right")

    if left_col and right_col:
        neighbors = {n.strip() for n in neighbors}
        mask = (
            (gdf[left_col].eq(country) & gdf[right_col].isin(neighbors))
            | (gdf[right_col].eq(country) & gdf[left_col].isin(neighbors))
        )
        gdf = gdf[mask].copy()
    else:
        print(
            "Warning: adm0_left/adm0_right columns not found; "
            "using all border lines in the file."
        )

    gdf = gdf[["geometry"]].dropna().copy()
    if gdf.empty:
        raise ValueError("Border lines are empty after filtering.")
    return gdf


def _clip_to_boundary(
    gdf: gpd.GeoDataFrame,
    boundary_shp: Optional[Path],
    epsg_m: int,
    buffer_m: float = 0.0,
) -> gpd.GeoDataFrame:
    if boundary_shp is None or not boundary_shp.exists():
        return gdf
    # Clean invalid/empty geometries before spatial ops
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    boundary = gpd.read_file(boundary_shp)
    boundary = boundary.to_crs(epsg=epsg_m)
    boundary = boundary[boundary.geometry.notna()].copy()
    boundary = boundary[~boundary.geometry.is_empty].copy()

    boundary_geom = _union_all(boundary.geometry)
    if buffer_m and buffer_m > 0:
        boundary_geom = boundary_geom.buffer(buffer_m)

    gdf = gdf.to_crs(epsg=epsg_m)

    # Try intersection; if it fails due to NaN/Inf, use safe clipping method
    try:
        intersect_mask = gdf.geometry.intersects(boundary_geom)
        gdf = gdf[intersect_mask].copy()
    except Exception as e:
        if "NaN" in str(e) or "Inf" in str(e):
            # Fall back to using bounds-based filtering
            boundary_bounds = boundary_geom.bounds
            bounds_mask = (
                (gdf.geometry.bounds.minx < boundary_bounds[2]) &
                (gdf.geometry.bounds.maxx > boundary_bounds[0]) &
                (gdf.geometry.bounds.miny < boundary_bounds[3]) &
                (gdf.geometry.bounds.maxy > boundary_bounds[1])
            )
            gdf = gdf[bounds_mask].copy()
        else:
            raise

    return gdf


def _load_farms(farms_path: Path) -> pd.DataFrame:
    if farms_path.suffix.lower() == ".pkl":
        return pd.read_pickle(farms_path)
    if farms_path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(farms_path)
    raise ValueError(f"Unsupported farms file type: {farms_path}")


def _build_farms_gdf(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    crs: str = "EPSG:4326",
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    base = df.copy()
    for col in (lat_col, lon_col):
        if col not in base.columns:
            raise KeyError(f"Missing column: {col}")
    mask = base[lat_col].notna() & base[lon_col].notna()
    gdf = gpd.GeoDataFrame(
        base.loc[mask].copy(),
        geometry=gpd.points_from_xy(
            base.loc[mask, lon_col],
            base.loc[mask, lat_col],
        ),
        crs=crs,
    )
    return base, gdf


def _compute_distances(
    farms_gdf: gpd.GeoDataFrame,
    border_gdf: gpd.GeoDataFrame,
    rivers_gdf: Optional[gpd.GeoDataFrame],
    epsg_m: int,
    river_buffer_m: float,
) -> gpd.GeoDataFrame:
    farms = farms_gdf.to_crs(epsg=epsg_m)
    border = border_gdf.to_crs(epsg=epsg_m)

    border_union = _union_all(border.geometry)
    farms["dist_border_m"] = farms.geometry.distance(border_union)

    # Default outputs
    farms["dist_border_river_m"] = np.nan
    farms["dist_border_dry_m"] = np.nan
    farms["nearest_border_type"] = np.nan

    if rivers_gdf is None or rivers_gdf.empty:
        return farms

    rivers = rivers_gdf.to_crs(epsg=epsg_m)
    river_union = _union_all(rivers.geometry)
    river_buffer = river_union.buffer(river_buffer_m)

    border = border.copy()
    border["is_river"] = border.geometry.intersects(river_buffer)
    border_river = border[border["is_river"]].copy()
    border_dry = border[~border["is_river"]].copy()

    if not border_river.empty:
        river_union_line = _union_all(border_river.geometry)
        farms["dist_border_river_m"] = farms.geometry.distance(river_union_line)

    if not border_dry.empty:
        dry_union_line = _union_all(border_dry.geometry)
        farms["dist_border_dry_m"] = farms.geometry.distance(dry_union_line)

    # Nearest type
    d_river = farms["dist_border_river_m"].to_numpy()
    d_dry = farms["dist_border_dry_m"].to_numpy()
    nearest = np.where(
        np.isfinite(d_river) & np.isfinite(d_dry),
        np.where(d_river <= d_dry, "river", "dry"),
        np.where(np.isfinite(d_river), "river", np.where(np.isfinite(d_dry), "dry", None)),
    )
    farms["nearest_border_type"] = nearest
    return farms


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Distance from farms to international border (river/dry optional)"
    )
    parser.add_argument(
        "--farms",
        default=str(CADASTRO_PKL),
        help="Path to farms data (pkl or csv)",
    )
    parser.add_argument(
        "--border-shp",
        default="",
        help="Shapefile for international border lines. If empty, uses Natural Earth when --auto-download is set.",
    )
    parser.add_argument(
        "--rivers-shp",
        default="",
        help="Shapefile for rivers/centerlines. If empty, uses Natural Earth when --auto-download is set.",
    )
    parser.add_argument(
        "--boundary-shp",
        default=str(RS_BOUNDARY_SHP),
        help="RS boundary shapefile used only for clipping",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        default=True,
        help="Download Natural Earth border/rivers data if not provided (default: on)",
    )
    parser.add_argument(
        "--no-auto-download",
        dest="auto_download",
        action="store_false",
        help="Disable auto-download of Natural Earth data",
    )
    parser.add_argument(
        "--neighbors",
        default="Uruguay,Argentina",
        help="Comma-separated neighbor countries to keep",
    )
    parser.add_argument(
        "--lat-col",
        default="LATITUDE_DECIMAL",
        help="Latitude column in farms data",
    )
    parser.add_argument(
        "--lon-col",
        default="LONGITUDE_DECIMAL",
        help="Longitude column in farms data",
    )
    parser.add_argument(
        "--id-col",
        default="COD_PROPRIEDADE_RURAL",
        help="Farm ID column for output merge",
    )
    parser.add_argument(
        "--epsg-m",
        type=int,
        default=31982,
        help="Projected CRS EPSG for distance in meters",
    )
    parser.add_argument(
        "--river-buffer-m",
        type=float,
        default=250.0,
        help="Buffer in meters to classify border as river-based",
    )
    parser.add_argument(
        "--clip-buffer-m",
        type=float,
        default=50000.0,
        help="Buffer in meters around RS boundary for clipping",
    )
    parser.add_argument(
        "--out",
        default=str(BORDER_PKL),
        help="Output pickle path",
    )
    args = parser.parse_args()

    farms_path = _expand(args.farms)
    out_path = _expand(args.out)
    boundary_shp = _expand(args.boundary_shp)

    border_shp = _expand(args.border_shp) if args.border_shp else None
    rivers_shp = _expand(args.rivers_shp) if args.rivers_shp else None

    if border_shp is None or not border_shp.exists():
        if args.auto_download:
            border_shp = _download_and_extract(NE_BORDER_URL, BORDER_DIR)
        else:
            raise FileNotFoundError(
                "Border shapefile not found. Provide --border-shp or use --auto-download."
            )

    if rivers_shp is None or not rivers_shp.exists():
        if args.auto_download:
            rivers_shp = _download_and_extract(NE_RIVERS_URL, HYDRO_DIR)
        else:
            rivers_shp = None

    neighbors = [n.strip() for n in args.neighbors.split(",") if n.strip()]

    farms_df = _load_farms(farms_path)
    if args.id_col not in farms_df.columns:
        raise KeyError(f"Missing farm id column: {args.id_col}")

    border_gdf = _load_border_lines(border_shp, country="Brazil", neighbors=neighbors)
    border_gdf = _clip_to_boundary(border_gdf, boundary_shp, args.epsg_m, args.clip_buffer_m)

    rivers_gdf = None
    if rivers_shp is not None:
        rivers_gdf = gpd.read_file(rivers_shp)
        rivers_gdf = _clip_to_boundary(
            rivers_gdf, boundary_shp, args.epsg_m, args.clip_buffer_m
        )

    base_df, farms_gdf = _build_farms_gdf(
        farms_df, lat_col=args.lat_col, lon_col=args.lon_col
    )

    farms_dist = _compute_distances(
        farms_gdf=farms_gdf,
        border_gdf=border_gdf,
        rivers_gdf=rivers_gdf,
        epsg_m=args.epsg_m,
        river_buffer_m=args.river_buffer_m,
    )

    # Merge distances back, preserving original row order
    base_df["dist_border_m"] = np.nan
    base_df["dist_border_river_m"] = np.nan
    base_df["dist_border_dry_m"] = np.nan
    base_df["nearest_border_type"] = None
    base_df.loc[farms_dist.index, "dist_border_m"] = farms_dist["dist_border_m"].values
    base_df.loc[farms_dist.index, "dist_border_river_m"] = farms_dist["dist_border_river_m"].values
    base_df.loc[farms_dist.index, "dist_border_dry_m"] = farms_dist["dist_border_dry_m"].values
    base_df.loc[farms_dist.index, "nearest_border_type"] = farms_dist["nearest_border_type"].values

    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_df[[args.id_col, "dist_border_m", "dist_border_river_m", "dist_border_dry_m", "nearest_border_type"]].to_pickle(out_path)

    print(f"Saved: {out_path}")
    print(base_df[["dist_border_m", "dist_border_river_m", "dist_border_dry_m"]].describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

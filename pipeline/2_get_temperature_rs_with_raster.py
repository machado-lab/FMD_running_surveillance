#=====================================================#
# get the temperature for each grid cell and other metrics
# for programs as  FMD
#=====================================================#
import sys
import pickle
import glob
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import ee
import geopandas as gpd
import requests
from datetime import date, timedelta
import time
import rasterio
from rasterio.mask import mask
import geemap
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from rasterio.sample import sample_gen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import (
    BASE_DIR, CREDENTIALS_DIR,
    CADASTRO_PKL, WEATHER_PKL, LST_DIR,
)

#clulate the survival
def fmd_survival_probability(T, t_days=30):

    # Fraction of virus in fast-decay compartment
    p = 0.85

    # Fast decay (dominant fraction)
    half_life_fast = 2.0 * np.exp(-0.20 * (T - 20))

    # Slow tail (protected fraction: soil, organic matter)
    half_life_slow = 10.0 * np.exp(-0.08 * (T - 20))

    # Avoid unrealistic values
    half_life_fast = np.maximum(half_life_fast, 0.05)
    half_life_slow = np.maximum(half_life_slow, 0.2)

    k1 = np.log(2) / half_life_fast
    k2 = np.log(2) / half_life_slow

    survival = (
        p * np.exp(-k1 * t_days) +
        (1 - p) * np.exp(-k2 * t_days)
    )

    return survival

# authenticate (first time opens browser) and  Initialize Earth Engine
# Set your own Google Cloud / Earth Engine project id via the GEE_PROJECT env var.
GEE_PROJECT = os.environ.get("GEE_PROJECT")
if not GEE_PROJECT:
    raise RuntimeError(
        "Set the GEE_PROJECT environment variable to your Google Earth Engine "
        "project id (e.g. `export GEE_PROJECT=my-ee-project`)."
    )
ee.Initialize(project=GEE_PROJECT)

# Locate client secret — place your OAuth JSON in credentials/
client_secrets = sorted(CREDENTIALS_DIR.glob("client_secret_*.json"))
if not client_secrets:
    raise FileNotFoundError(
        f"No client_secret_*.json found in {CREDENTIALS_DIR}. "
        "Download it from Google Cloud Console and place it there."
    )
gauth = GoogleAuth()
gauth.LoadClientConfigFile(str(client_secrets[0]))
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

LST_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load GAUL and ROI
# -------------------------
states = ee.FeatureCollection("FAO/GAUL/2015/level1")

rs = states.filter(
    ee.Filter.And(
        ee.Filter.eq("ADM0_NAME", "Brazil"),
        ee.Filter.eq("ADM1_NAME", "Rio Grande Do Sul")  # capital D and S
    )
)

roi = rs.geometry()

lst_col = ee.ImageCollection("MODIS/061/MOD11A1").select("LST_Day_1km")

def monthly_lst(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")

    img = (
        lst_col
        .filterDate(start, end)
        .mean()
        .multiply(0.02)
        .subtract(273.15)
        .clip(roi)
    )

    return img
# months list
today = datetime.date.today()
this_month_start = today.replace(day=1)
end_month = this_month_start - datetime.timedelta(days=1)

months = []
y, m = end_month.year, end_month.month

for _ in range(12):
    months.append((y, m))
    m -= 1
    if m == 0:
        m = 12
        y -= 1

months = months[::-1]
# Download locally
for year, month in months:

    img = monthly_lst(year, month)
    name = f"LST_{year}_{month:02d}_RS"
    out_file = LST_DIR / f"{name}.tif"

    # Skip if already downloaded
    if out_file.exists():
        print(f"Already exists, skipping: {out_file}")
        continue

    geemap.ee_export_image(
        img,
        filename=str(out_file),
        scale=1000,
        region=roi.bounds(),
        file_per_band=False
    )

    print("Downloaded:", out_file)
print("All downloads completed.")

#=====================================================#
#
#.  add meand temperature to cadastro and calutes riks
#
#=====================================================#
#load cadastro
with open(str(CADASTRO_PKL), "rb") as f:
    cadastro = pickle.load(f)
cadastro.head()

# ---- Sample raster value at a single lat/lon ----
def sample_raster(tif_path, lat, lon):
    """Returns the raster value at a given lat/lon, or NaN if outside bounds."""
    try:
        with rasterio.open(tif_path) as src:
            # Convert lat/lon to row/col pixel indices
            row, col = src.index(lon, lat)  # note: rasterio uses (x=lon, y=lat)
            # Check within bounds
            if 0 <= row < src.height and 0 <= col < src.width:
                val = src.read(1)[row, col]
                nodata = src.nodata
                if nodata is not None and val == nodata:
                    return np.nan
                if val <= -100:  # fallback for MODIS fill values
                    return np.nan
                return float(val)
            else:
                return np.nan
    except Exception:
        return np.nan

# ---- Loop over all monthly rasters ----
# Take exactly the last 12 months, sorted chronologically
tif_files = sorted(glob.glob(str(LST_DIR / "LST_*_RS.tif")))[-12:]
print(f"Using {len(tif_files)} rasters: {[os.path.basename(f) for f in tif_files]}")

coords = list(zip(cadastro["LONGITUDE_DECIMAL"], cadastro["LATITUDE_DECIMAL"]))

for tif_path in tif_files:
    parts = os.path.basename(tif_path).split("_")
    year, month = parts[1], parts[2]
    month_name = datetime.date(int(year), int(month), 1).strftime("%b_%Y")  # e.g. Mar_2025
    col_name = f"LST_{month_name}"

    with rasterio.open(tif_path) as src:
        vals = np.array([v[0] for v in src.sample(coords)])
        nodata = src.nodata
        vals = vals.astype(float)
        if nodata is not None:
            vals[vals == nodata] = np.nan
        vals[vals <= -100] = np.nan

    cadastro[col_name] = vals
    print(f"Done: {col_name}  |  valid: {cadastro[col_name].notna().sum()}/{len(cadastro)}")

# Verify exactly 12 new columns
lst_cols = [c for c in cadastro.columns if c.startswith("LST_")]
print(f"\nTotal LST columns: {len(lst_cols)}")
print(cadastro[lst_cols].head())
# get the survival probability
##############################

def fmd_survival_probability(T, t_days=30):
    """
    FMD survival using biphasic decay (aligned with soil survival literature)
    Based on monthly average temperature over a 30-day period

    Parameters
    https://www.sciencedirect.com/science/article/pii/S037811351100650X
    ----------
    T : float or array
        Monthly average temperature (°C)
    t_days : float
        Time period in days (default 30 for monthly calculation)

    Returns
    -------
    survival_prob : float or array
        Survival probability over the monthly period
    """
    # Fraction of virus in fast-decay compartment
    p = 0.85

    # Fast decay (dominant fraction)
    half_life_fast = 2.0 * np.exp(-0.20 * (T - 20))

    # Slow tail (protected fraction: soil, organic matter)
    half_life_slow = 10.0 * np.exp(-0.08 * (T - 20))

    # Avoid unrealistic values
    half_life_fast = np.maximum(half_life_fast, 0.05)
    half_life_slow = np.maximum(half_life_slow, 0.2)

    k1 = np.log(2) / half_life_fast
    k2 = np.log(2) / half_life_slow

    survival = (
        p * np.exp(-k1 * t_days) +
        (1 - p) * np.exp(-k2 * t_days)
    )

    return survival

lst_cols = [c for c in cadastro.columns if c.startswith("LST_")]

for col in lst_cols:
    suffix = col.replace("LST_", "")  # e.g. Mar_2025
    new_col = f"fmd_survival_prob_{suffix}"
    cadastro[new_col] = cadastro[col].apply(
        lambda t: fmd_survival_probability(t) if pd.notna(t) else np.nan
    )
    print(f"Done: {new_col}  |  valid: {cadastro[new_col].notna().sum()}/{len(cadastro)}")

# Calculate overall FMD survival probability (mean across all months)
surv_cols = [c for c in cadastro.columns if c.startswith("fmd_survival_prob_")]
cadastro["fmd_survival_prob_overall"] = cadastro[surv_cols].mean(axis=1)
print(f"Done: fmd_survival_prob_overall  |  valid: {cadastro['fmd_survival_prob_overall'].notna().sum()}/{len(cadastro)}")

# Verify
surv_cols = [c for c in cadastro.columns if c.startswith("fmd_survival_prob_")]

# Save pickle
with open(str(WEATHER_PKL), "wb") as f:
    pickle.dump(cadastro, f)
print(f"Saved: {WEATHER_PKL}")

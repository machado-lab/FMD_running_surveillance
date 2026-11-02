# =============================================================================
# FULL ROAD NETWORK — ONE-TIME DOWNLOAD & EXTRACT
# Builds raw_data/roads/roads_rs_full.shp (every drivable OSM highway class,
# not just major roads) for 3_1_road_animal_preassure.py and
# 3_get_spatial_net_features.py to consume.
#
# Source: Geofabrik's Brazil OSM extract (static file, downloaded once and
# cached locally), clipped to the official IBGE RS state boundary and filtered
# to drivable highway classes with pyrosm.
#
# WHY NOT osmnx/Overpass API: the live Overpass API can reject or time out
# requests depending on network/firewall conditions. A static PBF extract
# sidesteps the live query service entirely.
#
# WHY A SEPARATE SCRIPT / POSSIBLY SEPARATE PYTHON: this script needs
# `pyrosm`, which is not a dependency of the rest of the pipeline. If your
# system Python's TLS stack can't complete a handshake with Geofabrik, run
# this script with a Python that has a modern TLS stack and
# `pyrosm` + `geopandas` installed, e.g.:
#   python3 -m venv /path/to/venv_roads
#   /path/to/venv_roads/bin/pip install pyrosm geopandas shapely
#   /path/to/venv_roads/bin/python pipeline/3_0_download_full_roads.py
# Once raw_data/roads/roads_rs_full.shp exists, the rest of the pipeline
# (3_1_road_animal_preassure.py, 3_get_spatial_net_features.py) runs under the
# normal system python3 as before — no network access needed there.
# =============================================================================
import shutil
import sys
import urllib.request
from pathlib import Path

import geopandas as gpd
from pyrosm import OSM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import OSM_DIR, RS_BOUNDARY_SHP, ROADS_FULL_SHP

PBF_PATH = OSM_DIR / "brazil-latest.osm.pbf"
PBF_URL  = "https://download.geofabrik.de/south-america/brazil-latest.osm.pbf"

# Drivable OSM highway classes to include. Excludes foot/cycle-only classes
# (footway, cycleway, path, steps, pedestrian, bridleway, corridor, elevator,
# escalator, platform, proposed, construction, abandoned, bus_guideway, raceway)
# since livestock trucks cannot use them.
ROAD_HIGHWAY_INCLUDE = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "living_street", "service", "track", "road",
]


def download_pbf() -> None:
    if PBF_PATH.exists():
        print(f"  Reusing cached PBF: {PBF_PATH} "
              f"({PBF_PATH.stat().st_size / 1e9:.2f} GB)")
        return
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = PBF_PATH.with_suffix(".pbf.part")
    print(f"  Downloading {PBF_URL} ...")
    with urllib.request.urlopen(PBF_URL) as resp, open(tmp_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp_path.rename(PBF_PATH)
    print(f"  Saved {PBF_PATH} ({PBF_PATH.stat().st_size / 1e9:.2f} GB)")


def build_full_road_network() -> None:
    if not RS_BOUNDARY_SHP.exists():
        raise FileNotFoundError(f"RS boundary shapefile not found: {RS_BOUNDARY_SHP}")

    print("\nSTEP 1: Loading RS state boundary...")
    boundary = gpd.read_file(str(RS_BOUNDARY_SHP)).to_crs("EPSG:4326")
    polygon = boundary.geometry.union_all()
    print(f"  Boundary bounds: {polygon.bounds}")

    print("\nSTEP 2: Ensuring Brazil OSM extract is available...")
    download_pbf()

    print("\nSTEP 3: Extracting RS drivable road network from PBF "
          "(this can take several minutes for a whole-country file)...")
    osm = OSM(str(PBF_PATH), bounding_box=polygon)
    roads = osm.get_data_by_custom_criteria(
        custom_filter={"highway": ROAD_HIGHWAY_INCLUDE},
        filter_type="keep",
        keep_nodes=False,
        keep_ways=True,
        keep_relations=False,
    )
    if roads is None or roads.empty:
        raise RuntimeError("No road features extracted — check the PBF extract "
                            "and boundary polygon.")

    roads["highway"] = roads["highway"].apply(
        lambda v: v[0] if isinstance(v, list) else v
    )
    roads = roads[["geometry", "highway"]].dropna(subset=["geometry"]).copy()
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty]
    if roads.crs is None:
        roads = roads.set_crs("EPSG:4326")

    print(f"  Extracted {len(roads):,} road segments")
    print(f"    highway classes: {roads['highway'].value_counts().to_dict()}")

    print("\nSTEP 4: Saving shapefile...")
    ROADS_FULL_SHP.parent.mkdir(parents=True, exist_ok=True)
    roads.to_file(str(ROADS_FULL_SHP))
    print(f"  Saved -> {ROADS_FULL_SHP}")


if __name__ == "__main__":
    build_full_road_network()

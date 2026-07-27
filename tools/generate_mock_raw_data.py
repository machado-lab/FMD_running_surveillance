#!/usr/bin/env python3
"""
Generate fully synthetic mock raw data for public release.
================================================================================
This OVERWRITES the sensitive input files under raw_data/ with synthetic
content that matches the original schemas exactly, so the pipeline still runs
end to end — but contains NO real GPS coordinates, names, CPF/CNPJ, or any
other individual-identifying information.

What it replaces (sensitive):
  - raw_data/cadastro/*.csv
  - raw_data/gtas/{bovine,bubaline,ovi,capri,suine}_movs/*.csv

What it leaves untouched (public-safe):
  - roads, border, rs_boundary_ibge, hydro, LST temperature rasters,
    javali municipality survey.

Design notes:
  - Every file's REAL header/columns are read first, then only the contents are
    replaced — so schemas stay byte-for-byte compatible with the loaders.
  - A single synthetic "farm table" (id, IBGE, CPF, coords) drives both
    datasets so cross-file joins (cadastro <-> GTAS) stay consistent and the
    pipeline does not crash on empty joins.
  - Municipality (code, name) pairs come from geobr.
  - Coordinates use a dot decimal separator (as in the originals); area uses a
    Brazilian comma decimal separator (step 8 converts it).

Run:  python tools/generate_mock_raw_data.py
"""

import os
import sys
import glob
import random
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import (
    CADASTRO_DIR, GTAS_DIR,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
N_FARMS = 2500            # synthetic farms in the cadastro
N_MOV = {                 # movement records per GTAS species folder
    "bovine_movs": 1500,
    "bubaline_movs": 300,
    "ovi_movs": 1200,
    "capri_movs": 250,
    "suine_movs": 1200,
}
N_SLAUGHTERHOUSES = 20    # abattoir destinations (NOT farms; short codes < 6 digits)
SLAUGHTER_FRACTION = 0.25 # share of movements shipped to a slaughterhouse

# Rio Grande do Sul bounding box
LAT_MIN, LAT_MAX = -33.75, -27.05
LON_MIN, LON_MAX = -57.65, -49.70

# Species accepted by the cadastro loader (1_load_data.py)
CADASTRO_ESPECIES = [
    "Bovinos", "Ovino", "Suíno", "Caprinos", "Bubalinos",
    "Galinha", "Pato", "Peru", "Galinha de Angola",
]
# Species-specific total column per GTAS folder (see 1_load_data.py)
GTAS_TOT_COL = {
    "bovine_movs": "BOVINO_TOT",
    "bubaline_movs": "BUBALINO_TOT",
    "ovi_movs": "OVINO_TOT",
    "capri_movs": "CAPRINOS_TOT",
    "suine_movs": "SUINO_TOT",
}

rng = np.random.default_rng(SEED)
random.seed(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rs_municipalities():
    """Authoritative (IBGE code, name) pairs for RS, via geobr."""
    import geobr
    m = geobr.read_municipality(code_muni="RS", year=2020)
    m = m[["code_muni", "name_muni"]].dropna()
    m["code_muni"] = m["code_muni"].astype(int)
    return list(m.itertuples(index=False, name=None))  # [(code, name), ...]


def rand_coords(n):
    lat = rng.uniform(LAT_MIN, LAT_MAX, n).round(6)
    lon = rng.uniform(LON_MIN, LON_MAX, n).round(6)
    return lat, lon


def fake_cpfs(n):
    """Unique 11-digit numeric strings (first digit 1-9 so to_numeric is stable)."""
    seen, out = set(), []
    while len(out) < n:
        c = str(rng.integers(1, 10)) + "".join(
            str(d) for d in rng.integers(0, 10, 10)
        )
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def dmy(n, y0=2021, y1=2025):
    days = rng.integers(0, 365, n)
    years = rng.integers(y0, y1 + 1, n)
    base = pd.to_datetime(years.astype(str) + "-01-01")
    dt = base + pd.to_timedelta(days, unit="D")
    return dt.strftime("%d/%m/%Y")


def build_farm_table():
    munis = rs_municipalities()
    idx = rng.integers(0, len(munis), N_FARMS)
    codes = np.array([munis[i][0] for i in idx])
    names = np.array([munis[i][1] for i in idx])
    lat, lon = rand_coords(N_FARMS)
    farms = pd.DataFrame({
        "farm_id": [str(100000 + i) for i in range(N_FARMS)],
        "ibge": codes,
        "muni_name": names,
        "cpf": fake_cpfs(N_FARMS),
        "lat": lat,
        "lon": lon,
        "area": [f"{v:.1f}".replace(".", ",") for v in rng.uniform(1, 5000, N_FARMS)],
        "especie": rng.choice(CADASTRO_ESPECIES, N_FARMS),
        "qtd": rng.integers(1, 2000, N_FARMS),
    })
    return farms


def build_slaughterhouses(farms):
    """Abattoir destinations: short (< 6 digit) codes, deliberately NOT in the
    cadastro because slaughterhouses are not farms. Used by the slaughterhouse
    proximity step, which detects them via destination-code length.
    """
    lat, lon = rand_coords(N_SLAUGHTERHOUSES)
    return pd.DataFrame({
        "code": [str(5000 + i) for i in range(N_SLAUGHTERHOUSES)],  # 4 digits
        "ibge": rng.choice(farms["ibge"].values, N_SLAUGHTERHOUSES),
        "lat": lat,
        "lon": lon,
    })


# ---------------------------------------------------------------------------
# 1. Cadastro
# ---------------------------------------------------------------------------
def write_cadastro(farms):
    files = sorted(glob.glob(str(CADASTRO_DIR / "*.csv")))
    if not files:
        print("  (no cadastro CSV found — skipping)")
        return
    cols = list(pd.read_csv(files[0], sep=";", encoding="latin-1", nrows=0).columns)

    df = pd.DataFrame("", index=range(len(farms)), columns=cols)
    df["NOME_MUNICIPIO"] = farms["muni_name"].values
    df["COD_MUNICIPIO_IBGE"] = farms["ibge"].values
    df["PROPRIEDADE_RURAL"] = [f"FAZENDA_{i:05d}" for i in range(len(farms))]
    df["COD_PROPRIEDADE_RURAL"] = farms["farm_id"].values
    df["CPF_CNPJ_TITULAR"] = farms["cpf"].values
    df["AREA_PROPRIEDADE"] = farms["area"].values
    df["LATITUDE_DECIMAL"] = farms["lat"].values
    df["LONGITUDE_DECIMAL"] = farms["lon"].values
    df["ANO_DECLARACAO"] = rng.integers(2021, 2026, len(farms))
    df["DATA_DECLARACAO"] = dmy(len(farms))
    df["ESPECIE_ANIMAL"] = farms["especie"].values
    df["QTD_DECLARADA"] = farms["qtd"].values
    if "FINALIDADE_CRIACAO" in df.columns:
        df["FINALIDADE_CRIACAO"] = rng.choice(["Corte", "Leite", "Reprodução"], len(farms))

    # Collapse to a single file; clear the others so no stale real data remains.
    out = files[0]
    df.to_csv(out, sep=";", index=False, encoding="latin-1")
    for extra in files[1:]:
        df.iloc[0:0].to_csv(extra, sep=";", index=False, encoding="latin-1")
    print(f"  cadastro: {len(df):,} synthetic farms -> {Path(out).name}")


# ---------------------------------------------------------------------------
# 2. GTAS movements
# ---------------------------------------------------------------------------
def write_gtas(farms, slaughter):
    fid = farms["farm_id"].values
    fibge = farms.set_index("farm_id")["ibge"]
    flat = farms.set_index("farm_id")["lat"]
    flon = farms.set_index("farm_id")["lon"]
    sl = slaughter.set_index("code")

    for folder, n in N_MOV.items():
        files = sorted(glob.glob(str(GTAS_DIR / folder / "*.csv")))
        if not files:
            print(f"  (no CSV in {folder} — skipping)")
            continue
        cols = list(pd.read_csv(files[0], sep=";", encoding="latin-1", nrows=0).columns)

        # Origins are always real cadastro farms. Destinations are either another
        # cadastro farm or — for a fraction of records — a slaughterhouse.
        orig = rng.choice(fid, n)
        farm_dest = rng.choice(fid, n)
        is_sl = rng.random(n) < SLAUGHTER_FRACTION
        sl_dest = rng.choice(slaughter["code"].values, n)

        dest = np.where(is_sl, sl_dest, farm_dest)
        dest_ibge = np.where(is_sl, sl.loc[sl_dest, "ibge"].values, fibge.loc[farm_dest].values)
        dest_lat = np.where(is_sl, sl.loc[sl_dest, "lat"].values, flat.loc[farm_dest].values)
        dest_lon = np.where(is_sl, sl.loc[sl_dest, "lon"].values, flon.loc[farm_dest].values)

        df = pd.DataFrame("", index=range(n), columns=cols)
        df["SERIE"] = "MOCK"
        df["NUMERO"] = [str(1000000 + i) for i in range(n)]
        df["FINALIDADE"] = np.where(is_sl, "Abate",
                                    rng.choice(["Reprodução", "Engorda", "Outros"], n))
        df["DATA_EMISSAO"] = dmy(n)
        df["UF_ORIGEM"] = "RS"
        df["UF_DESTINO"] = "RS"
        df["COD_MUN_IBGE_ORIGEM"] = fibge.loc[orig].values
        df["COD_MUN_IBGE_DESTINO"] = dest_ibge
        df["COD_PROPRIEDADE_ORIGEM"] = orig
        df["COD_PROPRIEDADE_DESTINO"] = dest
        df["LATITUDE_DEC_DESTINO"] = dest_lat
        df["LONGITUDE_DEC_DESTINO"] = dest_lon
        tot_col = GTAS_TOT_COL[folder]
        if tot_col in df.columns:
            df[tot_col] = rng.integers(1, 500, n)

        out = files[0]
        df.to_csv(out, sep=";", index=False, encoding="latin-1")
        for extra in files[1:]:
            df.iloc[0:0].to_csv(extra, sep=";", index=False, encoding="latin-1")
        print(f"  gtas/{folder}: {n:,} movements -> {Path(out).name}")


# ---------------------------------------------------------------------------
def main():
    print("Generating synthetic mock raw data ...")
    farms = build_farm_table()
    slaughter = build_slaughterhouses(farms)
    write_cadastro(farms)
    write_gtas(farms, slaughter)
    print(f"  ({len(slaughter)} slaughterhouse destinations; movement origins/destinations "
          "are cadastro farms except abattoirs)")
    print("Done. Real GPS / names / CPF have been replaced with synthetic values.")


if __name__ == "__main__":
    main()

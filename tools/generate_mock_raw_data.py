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
  - raw_data/sisbravet/Relatorio_Ocorrencia_*.xlsx
  - raw_data/PNEFA_epicollec_bio_records/form-1__form-pvbr.csv

What it leaves untouched (public-safe):
  - roads, border, rs_boundary_ibge, hydro, LST temperature rasters,
    javali municipality survey.

Design notes:
  - Every file's REAL header/columns are read first, then only the contents are
    replaced — so schemas stay byte-for-byte compatible with the loaders.
  - A single synthetic "farm table" (id, IBGE, CPF, coords) drives all four
    datasets so cross-file joins (cadastro <-> GTAS <-> Epicollect <-> SISBRAVET)
    stay consistent and the pipeline does not crash on empty joins.
  - Municipality (code, name) pairs come from geobr, the same source step 6
    uses to resolve SISBRAVET município names to IBGE codes. This guarantees the
    CPF + município FMD label join yields a non-trivial positive class so the
    10-fold cross-validation in step 6 does not fail on a single-class target.
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
    CADASTRO_DIR, GTAS_DIR, SISBRAVET_DIR, EPICOLLECT_CSV,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
N_FARMS = 2500            # synthetic farms in the cadastro
N_VISITED = 250           # farms with an Epicollect biosecurity record
N_FMD_POSITIVE = 70       # visited farms flagged as FMD cases (>= 1 per CV fold)
N_MOV = {                 # movement records per GTAS species folder
    "bovine_movs": 1500,
    "bubaline_movs": 300,
    "ovi_movs": 1200,
    "capri_movs": 250,
    "suine_movs": 1200,
}
N_SISBRAVET_EXTRA = 150   # extra (non-matching) surveillance rows
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
# FMD-susceptible species accepted by the SISBRAVET loader (4_...py)
SISBRAVET_ESPECIES = ["Bovino", "Suíno", "Ovino", "Caprino"]

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
    """Authoritative (IBGE code, name) pairs for RS, via geobr.

    Same source step 6 uses to map município name -> IBGE code, so the FMD
    label join is guaranteed to resolve.
    """
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
# 3. SISBRAVET surveillance (FMD labels)
# ---------------------------------------------------------------------------
def write_sisbravet(farms):
    files = sorted(glob.glob(str(SISBRAVET_DIR / "*.xlsx")))
    if not files:
        print("  (no SISBRAVET xlsx found — skipping)")
        return
    cols = list(pd.read_excel(files[0], nrows=0).columns)

    # Positive cases: the first N_FMD_POSITIVE *visited* farms (so the Epicollect
    # training set has both classes). Same CPF + município as in the cadastro.
    pos = farms.iloc[:N_FMD_POSITIVE]
    n_pos = len(pos)
    n = n_pos + N_SISBRAVET_EXTRA

    lat, lon = rand_coords(n)
    df = pd.DataFrame("", index=range(n), columns=cols)
    df["Número da ocorrência"] = [str(900000 + i) for i in range(n)]
    df["UF"] = "RS"
    # First block = real matches; remainder = unrelated extras.
    muni = list(pos["muni_name"].values) + list(
        rng.choice(farms["muni_name"].values, N_SISBRAVET_EXTRA)
    )
    cpf = list(pos["cpf"].values) + fake_cpfs(N_SISBRAVET_EXTRA)
    df["Município"] = muni
    df["CPF"] = cpf
    df["Nome da propriedade"] = ""
    df["Latitude"] = [f"{v:.6f}".replace(".", ",") for v in lat]
    df["Longitude"] = [f"{v:.6f}".replace(".", ",") for v in lon]
    df["Espécie"] = rng.choice(SISBRAVET_ESPECIES, n)
    df["Situação"] = rng.choice(["Suspeita", "Confirmada", "Descartada"], n)
    df["Diagnóstico final"] = "Ausência de sinais clínicos"
    for c in ["Data de provável início", "Data da notificação", "Data do atendimento inicial"]:
        if c in df.columns:
            df[c] = dmy(n)
    for c in ["Número de susceptíveis", "Número de casos confirmados",
              "Número de casos prováveis", "Número de mortos",
              "Número de destruídos", "Número de abatidos"]:
        if c in df.columns:
            df[c] = rng.integers(0, 50, n)

    # Distribute across the existing file names (keeps the *_a_*.xlsx pattern).
    chunks = np.array_split(df.sample(frac=1, random_state=SEED).reset_index(drop=True), len(files))
    for path, chunk in zip(files, chunks):
        chunk.to_excel(path, index=False)
    print(f"  sisbravet: {n:,} rows ({n_pos} FMD-positive) across {len(files)} file(s)")


# ---------------------------------------------------------------------------
# 4. Epicollect biosecurity survey
# ---------------------------------------------------------------------------
def write_epicollect(farms):
    if not EPICOLLECT_CSV.exists():
        print("  (no Epicollect CSV found — skipping)")
        return
    cols = list(pd.read_csv(EPICOLLECT_CSV, encoding="utf-8-sig", nrows=0).columns)

    visited = farms.iloc[:N_VISITED]
    n = len(visited)
    df = pd.DataFrame("", index=range(n), columns=cols)
    if "ec5_uuid" in df.columns:
        df["ec5_uuid"] = [f"mock-{i:05d}" for i in range(n)]
    df["title"] = ""
    if "2_1_UF" in df.columns:
        df["2_1_UF"] = "RS"
    if "3_2_Cdigo_do_IBGE__N" in df.columns:
        df["3_2_Cdigo_do_IBGE__N"] = visited["ibge"].values
    df["4_3_Cdigo_do_Estabel"] = visited["farm_id"].values  # join key -> cadastro

    # Biosecurity features used by the model (6_...py). Count columns numeric,
    # the rest categorical Sim/Não so the ColumnTransformer has both types.
    count_cols = ["17_Nmero_de_Bovinos", "18_Nmero_de_Bubalino", "19_Nmero_de_Sunos",
                  "20_Nmero_de_Caprinos", "21_Nmero_de_Ovinos"]
    feature_cols = [
        '13_6_A_propriedade_t', '14_7_Havia_placa_de_', '15_8_Na_entrada_da_p',
        '22_9_H_diferentes_es', '23_10_Finalidade_Pri', '24_11_Qual_o_destino',
        '25_12_Tipo_de_criao_', '26_13_Grau_de_tecnif', '27_14_H_animais_conf',
        '28_15_O_pastopiquete', '29_16_H_fonte_de_gua', '30_17_A_propriedade_',
        '31_18_Com_que_frequn', '32_19_H_compartilham', '33_20_Com_que_frequn',
        '34_21_Aps_retornar_d', '35_22_Com_que_frequn', '36_23_Os_animais_rec',
        '37_24_Com_que_frequn', '38_25_Qual_o_mtodo_d', '39_26_H_fornecimento',
        '40_27_H_compartilham', '41_28_H_compartilham', '42_29_H_entrada_de_v',
        '43_30_H_compartilham', '44_31_A_propriedade_', '46_32_Ao_fazer_a_ins',
        '47_33_Ao_fazer_inspe', '48_34_Durante_a_sua_', '50_35_Com_qual_frequ',
        '51_36_O_Sr_a_conhece', '52_37_O_Sr_a_sabe_co', '54_38_Senhora_veteri',
        '55_Bovinos', '56_Bubalinos', '57_Sunos', '58_Caprinos', '59_Ovinos',
    ]
    for c in count_cols:
        if c in df.columns:
            df[c] = rng.integers(0, 500, n)
    for c in feature_cols:
        if c in df.columns:
            df[c] = rng.choice(["Sim", "Não"], n)

    df.to_csv(EPICOLLECT_CSV, index=False, encoding="utf-8-sig")
    print(f"  epicollect: {n} visited farms -> {EPICOLLECT_CSV.name}")


# ---------------------------------------------------------------------------
def main():
    print("Generating synthetic mock raw data ...")
    farms = build_farm_table()
    slaughter = build_slaughterhouses(farms)
    write_cadastro(farms)
    write_gtas(farms, slaughter)
    write_sisbravet(farms)
    write_epicollect(farms)
    print(f"  ({len(slaughter)} slaughterhouse destinations; movement origins/destinations "
          "are cadastro farms except abattoirs)")
    print("Done. Real GPS / names / CPF have been replaced with synthetic values.")


if __name__ == "__main__":
    main()

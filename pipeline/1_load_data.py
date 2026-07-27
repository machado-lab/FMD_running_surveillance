# Load raw data from raw_data/cadastro and raw_data/gtas
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import (
    CADASTRO_DIR, GTAS_DIR,
    CADASTRO_PKL, MOVEMENTS_PKL,
)

# ===============================================================#
# Loading cadastro dataset -----
# ===============================================================#

def read_csv_with_fallback(path):
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    # First pass: auto-detect delimiter with Python engine
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            continue
    # Second pass: try common delimiters explicitly
    for enc in encodings:
        for sep in (",", ";", "\t", "|"):
            try:
                return pd.read_csv(path, encoding=enc, sep=sep, engine="python")
            except UnicodeDecodeError:
                break
            except pd.errors.ParserError:
                continue
    # Last resort: skip bad lines
    return pd.read_csv(path, encoding="latin-1", sep=None, engine="python", on_bad_lines="skip")


def load_data_from(path):
    files = os.listdir(path)
    dataframes = []
    for file in files:
        if file.lower().endswith(".csv"):
            df = read_csv_with_fallback(os.path.join(path, file))
            dataframes.append(df)
    if not dataframes:
        return pd.DataFrame()
    return pd.concat(dataframes, ignore_index=True)


def load_data():
    return load_data_from(str(CADASTRO_DIR))


# ===============================================================#
# Species grouping configuration -----
# ===============================================================#
# Maps each species subfolder → (species_group label, species-specific total column)
#
# Grouping logic:
#   bovine_movs  + bubaline_movs  → bovine_bubalinos  (ruminants, FMD-susceptible)
#   ovi_movs     + capri_movs     → ovinos_caprinos   (small ruminants, FMD-susceptible)
#   suine_movs                    → suinos            (NOT grouped with ruminants)
GTAS_SPECIES_FOLDERS = {
    "bovine_movs":   ("bovine_bubalinos", "BOVINO_TOT"),
    "bubaline_movs": ("bovine_bubalinos", "BUBALINO_TOT"),
    "ovi_movs":      ("ovinos_caprinos",  "OVINO_TOT"),
    "capri_movs":    ("ovinos_caprinos",  "CAPRINOS_TOT"),
    "suine_movs":    ("suinos",           "SUINO_TOT"),
}

# Columns shared across all species files
GTAS_COMMON_COLS = [
    "SERIE", "NUMERO", "FINALIDADE", "DATA_EMISSAO", "HORA_EMISSAO",
    "LANCAMENTO", "MEIO_TRANSPORTE",
    "UF_ORIGEM", "COORD_REGIONAL_ORIGEM", "IDA_ORIGEM", "MUNICIPIO_ORIGEM",
    "COD_MUN_IBGE_ORIGEM", "LOCALIDADE_ORIGEM",
    "LATITUDE_DEC_ORIGEM", "LONGITUDE_DEC_ORIGEM",
    "PROP_RURAL_ORIGEM", "COD_PROPRIEDADE_ORIGEM", "PRODUTOR_ORIGEM",
    "TIPO_PRODUTOR_ORIGEM", "ESFERA_ESTAB_ORIGEM",
    "UF_DESTINO", "COORD_REGIONAL_DESTINO", "IDA_DESTINO", "MUNICIPIO_DESTINO",
    "COD_MUN_IBGE_DESTINO", "LOCALIDADE_DESTINO",
    "LATITUDE_DEC_DESTINO", "LONGITUDE_DEC_DESTINO",
    "PROP_RURAL_DESTINO", "COD_PROPRIEDADE_DESTINO", "PRODUTOR_DESTINO",
    "TIPO_PRODUTOR_DESTINO", "ESFERA_ESTAB_DESTINO",
]


def load_gtas_movements(base_path):
    """
    Load GTAS movement records from all species subfolders and return a
    unified, standardized DataFrame.

    All species-specific logic is centralized here. Downstream scripts
    must use the output columns species_group and animal_total only.

    Parameters
    ----------
    base_path : str or Path
        Path to the raw_data/gtas directory.

    Returns
    -------
    pd.DataFrame
        Unified movements dataset with:
          - All shared GTAS fields (date, origin/destination properties, etc.)
          - species_group : 'bovine_bubalinos' | 'ovinos_caprinos' | 'suinos'
          - animal_total  : total animals moved per record (species-group level)
    """
    base_path = Path(base_path)
    all_frames = []

    for folder, (species_group, tot_col) in GTAS_SPECIES_FOLDERS.items():
        folder_path = base_path / folder
        if not folder_path.is_dir():
            print(f"Warning: folder not found, skipping — {folder_path}")
            continue

        csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]
        if not csv_files:
            print(f"Warning: no CSV files found in {folder_path}")
            continue

        raw_frames = []
        for fname in csv_files:
            df = read_csv_with_fallback(str(folder_path / fname))
            raw_frames.append(df)

        species_df = pd.concat(raw_frames, ignore_index=True)

        # Keep shared columns that exist in this file (handles minor schema differences)
        keep_cols = [c for c in GTAS_COMMON_COLS if c in species_df.columns]
        result = species_df[keep_cols].copy()

        # Normalize species-specific total to the standard animal_total column
        if tot_col in species_df.columns:
            result["animal_total"] = pd.to_numeric(species_df[tot_col], errors="coerce")
        else:
            print(f"Warning: expected column '{tot_col}' not found in {folder}, animal_total set to NA")
            result["animal_total"] = pd.NA

        result["species_group"] = species_group
        all_frames.append(result)
        print(f"  Loaded {folder}: {len(result):,} records → species_group='{species_group}'")

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


# ===============================================================#
# Main: load, process and save both datasets -----
# ===============================================================#
if __name__ == "__main__":

    # ----------------------------------------------------------------
    # 1. Cadastro
    # ----------------------------------------------------------------
    df_all = load_data()
    if df_all.empty:
        raise ValueError("No CSV files loaded from raw_data/cadastro")

    df_all["DATA_DECLARACAO"] = pd.to_datetime(
        df_all["DATA_DECLARACAO"], errors="coerce", dayfirst=True
    )

    # Keep only the latest declaration per property
    df_latest = (
        df_all.sort_values("DATA_DECLARACAO")
        .groupby("COD_PROPRIEDADE_RURAL", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    df_latest = df_latest[[
        "COD_MUNICIPIO_IBGE",
        "COD_PROPRIEDADE_RURAL", "AREA_PROPRIEDADE",
        "LATITUDE_DECIMAL", "LONGITUDE_DECIMAL",
        "ESPECIE_ANIMAL", "QTD_DECLARADA",
    ]]

    especies_of_interests = [
        "Bovinos", "Ovino", "Suíno", "Caprinos", "Bubalinos",
        "Galinha", "Pato", "Peru", "Galinha de Angola",
    ]
    df_latest2 = df_latest[df_latest["ESPECIE_ANIMAL"].isin(especies_of_interests)]

    # ----------------------------------------------------------------
    # 2. Movements (GTAS) — all species
    # ----------------------------------------------------------------
    print("\nLoading GTAS movements ...")
    movements = load_gtas_movements(GTAS_DIR)
    if movements.empty:
        raise ValueError("No movement records loaded from raw_data/gtas")

    # Subset to columns needed by downstream scripts
    # species_group and animal_total replace all species-specific columns
    keep_movements = [
        c for c in [
            "DATA_EMISSAO", "FINALIDADE",
            "LATITUDE_DEC_DESTINO", "LONGITUDE_DEC_DESTINO",
            "COD_MUN_IBGE_ORIGEM", "COD_PROPRIEDADE_ORIGEM",
            "COD_PROPRIEDADE_DESTINO", "COD_MUN_IBGE_DESTINO",
            "species_group", "animal_total",
        ]
        if c in movements.columns
    ]
    movements2 = movements[keep_movements].copy()

    # ----------------------------------------------------------------
    # 3. Save
    # ----------------------------------------------------------------
    movements2.to_pickle(str(MOVEMENTS_PKL))
    df_latest2.to_pickle(str(CADASTRO_PKL))

    print(f"\nmovements2 saved: {len(movements2):,} rows → {MOVEMENTS_PKL}")
    print(movements2["species_group"].value_counts().to_string())
    print(f"\ncadastro saved  : {len(df_latest2):,} rows → {CADASTRO_PKL}")

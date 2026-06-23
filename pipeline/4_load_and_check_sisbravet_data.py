import sys
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import SISBRAVET_DIR, INTERMEDIATE, SISBRAVET_PKL

WORKOUT_DATA = INTERMEDIATE
WORKOUT_DATA.mkdir(exist_ok=True)

# ============================================================================
# 1. LOAD ALL EXCEL FILES
# ============================================================================
def load_sisbravet_data(data_path):
    """Load and concatenate all sisbravet Excel files"""
    excel_files = sorted(data_path.glob("*.xlsx"))
    print(f"Found {len(excel_files)} Excel files:")
    for f in excel_files:
        print(f"  - {f.name}")

    dfs = []
    for file in excel_files:
        try:
            df = pd.read_excel(file)
            print(f"✓ Loaded {file.name}: {len(df)} rows, {len(df.columns)} columns")
            dfs.append(df)
        except Exception as e:
            print(f"✗ Error loading {file.name}: {e}")

    data = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"\n{'='*60}\nTOTAL DATA: {len(data)} rows, {len(data.columns)} columns\n{'='*60}\n")
    return data
df = load_sisbravet_data(SISBRAVET_DIR)

# ============================================================================
# 1.5 FILTER FOR FMD-SUSCEPTIBLE SPECIES
# ============================================================================
# FMD-susceptible species: Cattle, Pigs, Sheep, Goats
df['Espécie'].unique()
fmd_susceptible = ['Bovino', 'Suíno', 'Ovino', 'Caprino']
df = df[df['Espécie'].isin(fmd_susceptible)].copy()

# ============================================================================
# ============================================================================
# 3. COORDINATE CONVERSION & SPATIAL ANALYSIS
# ============================================================================

def convert_coordinate(coord_str):
    """Convert Brazilian format (comma as decimal) to standard float"""
    if pd.isna(coord_str):
        return np.nan
    try:
        return float(str(coord_str).replace(',', '.'))
    except:
        return np.nan

df['Latitude_num'] = df['Latitude'].apply(convert_coordinate)
df['Longitude_num'] = df['Longitude'].apply(convert_coordinate)

# Convert dates early for spatial analysis
date_cols = ['Data de provável início', 'Data da notificação', 'Data do atendimento inicial']
for col in date_cols:
    df[col + '_dt'] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')

df['Ano'] = df['Data da notificação_dt'].dt.year
df['Mes'] = df['Data da notificação_dt'].dt.month
mun_with_coords = df[df['Latitude_num'].notna() & df['Longitude_num'].notna()].groupby('Município').agg({
    'Latitude_num': 'mean',
    'Longitude_num': 'mean',
    'Número da ocorrência': 'count'
}).rename(columns={'Número da ocorrência': 'Occurrences'}).sort_values('Occurrences', ascending=False)

# ============================================================================
# 4. TEMPORAL ANALYSIS
# ============================================================================

df['Ano_Mes'] = df['Data da notificação_dt'].dt.to_period('M')

print(f"\nOccurrences by Year:")
print(df['Ano'].value_counts().sort_index().to_string())

print(f"\n\nOccurrences by Month:")
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_counts = df['Mes'].value_counts().sort_index()
for month, count in month_counts.items():
    print(f"  {month_names[month-1]}: {count}")

# ============================================================================
# 5. SPECIES AND DISEASE ANALYSIS
# ============================================================================
print(f"\n{'='*60}\nSPECIES AND DISEASE ANALYSIS\n{'='*60}")

print(f"\nFMD-Susceptible Species distribution:")
print(df['Espécie'].value_counts().to_string())

print(f"\n\nTop 15 Diagnosis results (BEFORE cleaning):")
print(df['Diagnóstico final'].value_counts().head(15).to_string())

# ============================================================================
# DIAGNOSIS FILTERING FOR FMD-SUSCEPTIBLE SPECIES
# ============================================================================
print(f"\n{'='*60}\nDIAGNOSIS FILTERING FOR FMD RELEVANCE\n{'='*60}")

# Keywords indicating FMD-susceptible species diseases (KEEP these)
keep_keywords = [
    # FMD-specific
    "febre aftosa",

    # Cattle/Bovine diseases
    "tuberculose", "tuberculosis", "brucelose", "brucella",
    "mastite", "diarréia viral bovina", "diarrhea viral",
    "anaplasmose", "anemia infecciosa", "babesiose", "babesia",
    "antraz", "carbúnculo", "campilobacteriose",
    "leucose", "listeriose", "paratuberculose", "pleuropneumonia",
    "tétano", "tripanosomose",

    # Swine diseases
    "aujeszky", "pseudoraiva", "parvovirose suína", "parvovirus",
    "influenza dos suínos", "febre suína",

    # Sheep/Goat diseases
    "sarna ovina", "sarna", "maedi-visna", "epididimite ovina", "brucella ovis",

    # Multi-species diseases
    "raiva", "leptospirose", "clostridiose", "rinotraqueíte", "rinotracheitis",

    # Valid findings for susceptible species (NO signs of disease or non-infectious)
    "ausência de sinais", "absence of signs",
    "nao se aplica", "não se aplica", "not applicable",
    "agravo não infeccioso", "non-infectious",
    "sem diagnóstico conclusivo", "inconclusive"
]

# Keywords indicating ONLY non-FMD-susceptible species (REMOVE only if EXCLUSIVELY these)
equine_only_keywords = [
    "mormo", "garrotilho", "arterite viral equina",
    "rinopneumonia equina", "influenza equina"
]

poultry_only_keywords = [
    "newcastle", "gallinarum", "gumboro", "marek", "mycoplasma synoviae",
    "influenza aviária", "influenza das aves"
]

wildlife_only_keywords = [
    "febre do nilo", "nipah", "encefalite japonesa"
]

def is_fmd_susceptible_diagnosis(diagnosis):
    """
    Filter diagnoses for FMD-susceptible species (Bovino, Suíno, Ovino, Caprino)

    KEEP if:
      1. Contains FMD-susceptible species disease keywords
      2. Mentions bovine/swine/sheep/goat species names
      3. Is inconclusive/non-diagnostic

    REMOVE ONLY if:
      - Contains EXCLUSIVELY equine-specific OR poultry-specific diseases
      - AND does NOT mention susceptible species
    """
    if pd.isna(diagnosis):
        return True

    text = diagnosis.lower()

    # STEP 1: KEEP if contains FMD-susceptible disease keywords
    if any(k.lower() in text for k in keep_keywords):
        return True

    # STEP 2: KEEP if explicitly mentions susceptible species
    susceptible_species = ["bovina", "bovino", "cattle",
                          "suína", "suino", "porcino", "swine",
                          "ovina", "ovino", "sheep",
                          "caprino", "caprini", "goat"]
    if any(sp.lower() in text for sp in susceptible_species):
        return True

    # STEP 3: REMOVE if contains ONLY equine/poultry/wildlife diseases
    has_equine = any(k.lower() in text for k in equine_only_keywords)
    has_poultry = any(k.lower() in text for k in poultry_only_keywords)
    has_wildlife = any(k.lower() in text for k in wildlife_only_keywords)

    # Remove only if has exclusive non-susceptible disease AND no susceptible mentions
    if (has_equine or has_poultry or has_wildlife):
        return False

    # STEP 4: DEFAULT - KEEP (conservative for epidemiology)
    # Unknown diagnoses in susceptible animals are kept for further investigation
    return True

# Apply filter
print(f"\nOriginal records: {len(df)}")
df_clean = df[df["Diagnóstico final"].apply(is_fmd_susceptible_diagnosis)].copy()
records_removed = len(df) - len(df_clean)

print(f"Records after filtering: {len(df_clean)}")
print(f"Records removed: {records_removed} ({records_removed/len(df)*100:.1f}%)")

if records_removed > 0:
    removed_diagnoses = df[~df["Diagnóstico final"].apply(is_fmd_susceptible_diagnosis)]['Diagnóstico final'].value_counts()
    print(f"\nRemoved diagnoses (top 10):")
    for diag, count in removed_diagnoses.head(10).items():
        print(f"  {diag}: {count} records")

# Use filtered dataframe
df = df_clean

print(f"\n\nTop 15 Diagnosis results (AFTER cleaning):")
print(df['Diagnóstico final'].value_counts().head(15).to_string())



# ============================================================================
# 6. CASE STATUS AND ANIMAL IMPACT
# ============================================================================
print(f"\n{'='*60}\nCASE STATUS AND ANIMAL IMPACT\n{'='*60}")

print(f"\nCase Status:")
print(df['Situação'].value_counts().to_string())

impact_cols = ['Número de susceptíveis', 'Número de casos confirmados',
               'Número de casos prováveis', 'Número de mortos',
               'Número de destruídos', 'Número de abatidos']

print(f"\n\nTotal animals by impact category:")
for col in impact_cols:
    total = df[col].sum()
    print(f"  {col}: {int(total):,}")

# ============================================================================
# Save combined data as pickle file
with open(str(SISBRAVET_PKL), 'wb') as f:
    pickle.dump(df, f)
print(f"\n✓ Combined FMD data saved to: {SISBRAVET_PKL}")

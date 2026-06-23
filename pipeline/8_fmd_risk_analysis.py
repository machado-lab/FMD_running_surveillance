# ==============================================================================
# FMD FARM-LEVEL RISK ANALYSIS
# ==============================================================================
# Purpose:
#   Produce two interpretable, composite FMD risk scores for every farm
#   in the state of Rio Grande do Sul (RS), Brazil. Note: the input
#   dataset is already filtered to FMD-susceptible species in
#   7_combine_and_check_data.py (Bovinos, Suíno, Caprinos, Bubalinos).
#       1. Equal-weight risk score  — all components weighted 1/9
#       2. Epidemiologically weighted risk score — literature-grounded weights
#
# Input:
#   intermediate/banco_final_to_risk_analysis.pkl
#
# Outputs (all saved to results/):
#   fmd_risk_scored.csv            — full scored dataset (primary deliverable)
#
# Key assumptions (documented inline at each step):
#   - network_risk_score NaN → median imputation (N printed at runtime)
#   - fmd_survival_prob_overall: HIGHER = more risk
#         (FMD virus environmental survival probability, temperature-driven)
#   - biosecurity_farm_score:   HIGHER = more protected → INVERTED for risk
#   - dist_border_m:            CLOSER = higher risk    → INVERTED for risk
#   - road_exposure_risk_raw:   HIGHER exposure = higher risk (direct, no inversion)
#   - nearest_border_type:      only "dry" + missing    → EXCLUDED
#   - javali_risk_smooth_iter:  higher = higher wildlife reservoir risk
#   - slaughterhouse_risk:      higher = more slaughterhouse traffic = higher risk (direct)
#
# Style:
#   Linear, fully sequential code. No custom classes or multi-use functions.
#   Heavily commented for reproducibility and publication readiness.
#   Libraries: pandas, numpy, matplotlib, seaborn only.
#
# Author : ncesped
# Date   : 2026-04-03
# ==============================================================================

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import BANCO_FINAL_PKL, RESULTS

# ==============================================================================
# SECTION 0 — PATHS AND OUTPUT FOLDER
# ==============================================================================

OUTPUT_DIR = RESULTS
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[0] Output directory ready: {OUTPUT_DIR}\n")


# ==============================================================================
# SECTION 1 — LOAD DATA
# ==============================================================================
combined_df = pd.read_pickle(str(BANCO_FINAL_PKL))

print(f"    Shape            : {combined_df.shape}")
print(f"    Columns          : {list(combined_df.columns)}")

# ==============================================================================
# SECTION 2 — DATA CLEANING
# ==============================================================================
# Confirm required columns are present (dataset updated in 7_combine_and_check_data.py)
required_cols = [
    "COD_PROPRIEDADE_RURAL",
    "AREA_PROPRIEDADE",
    "ESPECIE_ANIMAL",
    "QTD_DECLARADA",
    "in_sisbravet",
    "network_risk_score",
    "n_neighbors_5km",
    "animals_within_5km",
    "road_exposure_risk_raw",
    "fmd_survival_prob_overall",
    "dist_border_m",
    "nearest_border_type",
    "biosecurity_farm_score",
    "javali_risk_smooth_iter",
    "slaughterhouse_risk"
]
missing_cols = [c for c in required_cols if c not in combined_df.columns]
assert not missing_cols, f"Missing required columns: {missing_cols}"

print("\n    FMD susceptible species in dataset:")
print(combined_df["ESPECIE_ANIMAL"].value_counts().to_string())

# Keep only analysis-relevant columns (avoid leaking IDs like CPF_CNPJ_TITULAR)
analysis_cols = [
    "COD_PROPRIEDADE_RURAL",
    "AREA_PROPRIEDADE",
    "ESPECIE_ANIMAL",
    "QTD_DECLARADA",
    "in_sisbravet",
    "network_risk_score",
    "n_neighbors_5km",
    "animals_within_5km",
    "road_exposure_risk_raw",
    "fmd_survival_prob_overall",
    "dist_border_m",
    "nearest_border_type",
    "biosecurity_farm_score",
    "javali_risk_smooth_iter",
    "slaughterhouse_risk",
]
combined_df = combined_df[analysis_cols].copy()

# --------------------------------------------------------------------------
# 2a. Fix AREA_PROPRIEDADE: Brazilian decimal separator is a comma
#     e.g. '15,8' → 15.8   '170' → 170.0
# --------------------------------------------------------------------------
combined_df['AREA_PROPRIEDADE'] = (
    combined_df['AREA_PROPRIEDADE']
    .astype(str)
    .str.strip()
    .str.replace(',', '.', regex=False)
    .pipe(pd.to_numeric, errors='coerce')   # non-parseable → NaN
)
print(f"    AREA_PROPRIEDADE — NaN after conversion: "
      f"{combined_df['AREA_PROPRIEDADE'].isna().sum()}")

# Ensure javali risk is numeric (should already be float, but enforce for safety)
combined_df['javali_risk_smooth_iter'] = pd.to_numeric(
    combined_df['javali_risk_smooth_iter'], errors='coerce'
)

# --------------------------------------------------------------------------
# 2b. Impute network_risk_score with the column median
#
# ASSUMPTION: NaN means the farm had no recorded live-animal movement
#   events in the movement database — not that it is risk-free. We use
#   the median rather than zero, which would systematically under-estimate
#   risk for farms that are simply absent from the movement registry.
#   Median imputation is a conservative, widely accepted approach for
#   missing network data in spatial-epidemiology contexts.
# --------------------------------------------------------------------------
network_median = combined_df['network_risk_score'].median()
n_network_nan  = combined_df['network_risk_score'].isna().sum()
pct_missing    = n_network_nan / len(combined_df) * 100

combined_df['network_risk_score'] = combined_df['network_risk_score'].fillna(
    network_median
)
print(f"    network_risk_score — {n_network_nan:,} NaN ({pct_missing:.1f} %) "
      f"imputed with median = {network_median:.4f}")

# --------------------------------------------------------------------------
# 2c. Median-impute remaining numeric columns with small NaN counts
#     (n_neighbors_5km, animals_within_5km, dist_3_roads_sum_m_div_3,
#      fmd_survival_prob_overall, dist_border_m, AREA_PROPRIEDADE,
#      QTD_DECLARADA, javali_risk_smooth_iter)
#
#     Median is preferred over mean because several variables
#     (animals_within_5km, n_neighbors_5km) are right-skewed.
# --------------------------------------------------------------------------
cols_to_impute = [
    'n_neighbors_5km',
    'animals_within_5km',
    'road_exposure_risk_raw',
    'fmd_survival_prob_overall',
    'dist_border_m',
    'AREA_PROPRIEDADE',
    'QTD_DECLARADA',
    'javali_risk_smooth_iter',
    'slaughterhouse_risk',
]

for col in cols_to_impute:
    n_nan = combined_df[col].isna().sum()
    if n_nan > 0:
        med = combined_df[col].median()
        combined_df[col] = combined_df[col].fillna(med)
        print(f"    {col:<35s} — {n_nan:,} NaN imputed with median = {med:.4f}")

# --------------------------------------------------------------------------
# 2d. Note on nearest_border_type: all non-missing rows have the value "dry"
#     (confirmed in data inspection). This variable has effectively zero
#     variation and therefore no discriminatory power. We exclude it from
#     the risk model. Border risk will be captured by dist_border_m alone.
# --------------------------------------------------------------------------
border_type_counts = combined_df['nearest_border_type'].value_counts(dropna=False)
print(f"\n    nearest_border_type distribution (all rows are 'dry' — EXCLUDED):")
print(f"    {border_type_counts.to_dict()}")

# Final check
print("\n    Missing values after cleaning:")
print(combined_df.isnull().sum().to_string())
print(f"\n    Final shape after cleaning: {combined_df.shape}")


# ==============================================================================
# SECTION 3 — FEATURE ENGINEERING: RISK COMPONENTS
# ==============================================================================
# Each component is stored as a separate column.
# The sign convention is: HIGHER raw value = HIGHER risk.
# Columns that violate this (accessibility, border, biosecurity) are
# flagged here and inverted AFTER normalization in Section 4.
#
# Keeping components separate makes weights trivially editable and
# allows individual components to be audited in the final dataset.



# --- Component 1: Spatial density risk ---
# Higher farm count within 5 km = denser local population = more contact
# opportunities = higher local amplification risk.
# Source: n_neighbors_5km (number of farms within 5 km radius)
combined_df['comp_spatial_density'] = combined_df['n_neighbors_5km']

# --- Component 2: Animal pressure risk ---
# Total number of susceptible animals within 5 km radius.
# More animals = larger susceptible host pool = higher transmission pressure
# (Keeling & Rohani, 2008 — mass-action transmission assumption).
# Source: animals_within_5km
combined_df['comp_animal_pressure'] = combined_df['animals_within_5km']

# --- Component 3: Network / movement risk ---
# Captures live-animal movement pathways through trade networks.
# FMD spreads primarily via movement of infected animals and contaminated
# vehicles/fomites (Bigras-Poulin et al., 2006). Higher score = more
# connected farm = faster potential spread.
# Source: network_risk_score (already median-imputed above)
combined_df['comp_network_risk'] = combined_df['network_risk_score']

# --- Component 4: Road exposure risk ---
# ASSUMPTION: Farms near roads with high animal traffic face greater risk of
#   exposure to fomites carried by livestock transport vehicles. This component
#   combines animal flow on nearby roads with farm-to-road distance via an
#   exponential decay (Nigsch et al., 2013; Garner & Lack, 1995; Boklund et
#   al., 2009). The decay parameter λ = ln(2)/6000 m⁻¹ sets the half-life at
#   2 km per road segment, consistent with reported vehicle-borne spread zones.
# Direction: HIGHER value = HIGHER risk → direct (no inversion needed).
# Source: road_exposure_risk_raw (computed in 7_combine_and_check_data.py)
combined_df['comp_road_exposure'] = combined_df['road_exposure_risk_raw']

# --- Component 5: Border risk ---
# ASSUMPTION: Proximity to the international border with Argentina and
#   Uruguay raises the risk of exotic-strain reintroduction via informal
#   animal movements, contaminated trucks, or illegal trade (Lyons et al.,
#   2015). Rio Grande do Sul is Brazil's only state bordering both countries.
# Direction: SMALLER distance = HIGHER risk  → will be INVERTED in Section 4.
# Note: nearest_border_type excluded (uniform "dry", no variation).
# Source: dist_border_m
combined_df['comp_border'] = combined_df['dist_border_m']

# --- Component 6: Wildlife (javali) reservoir risk ---
# ASSUMPTION: Higher wild boar (javali) risk increases the probability of
#   wildlife-mediated FMD introduction and local persistence. This layer
#   is municipality-level smoothed risk, merged to farms by COD_MUNICIPIO_IBGE.
# Direction: HIGHER value = HIGHER risk (direct, no inversion needed).
# Source: javali_risk_smooth_iter
combined_df['comp_wildlife_javali'] = combined_df['javali_risk_smooth_iter']

# --- Component 7: Biosecurity risk (inverse) ---
# ASSUMPTION: Higher biosecurity_farm_score indicates more protective
#   measures in place (fencing quality, access control, disinfection
#   protocols) — i.e. LOWER transmission probability.
# Direction: HIGHER score = LOWER risk → will be INVERTED in Section 4.
# Source: biosecurity_farm_score (range ~0.026–0.83)
combined_df['comp_biosecurity'] = combined_df['biosecurity_farm_score']

# --- Component 8: Epidemiological susceptibility ---
# ASSUMPTION: fmd_survival_prob_overall is the probability that the FMD
#   virus survives in the ambient environment at the farm's location.
#   This is computed from local temperature/humidity data (see
#   weather_probs_fmd.pkl and plot_fmd_survival_virus_temperature.R).
#   Higher probability = virus persists longer = greater risk of local
#   transmission and amplification (Sanson, 1994).
# Direction: HIGHER value = HIGHER risk (direct, no inversion needed).
# Source: fmd_survival_prob_overall (range ~0.48–0.83)
combined_df['comp_fmdv_survival_prob_overall'] = combined_df['fmd_survival_prob_overall']

# --- Component 9: Slaughterhouse proximity risk ---
# ASSUMPTION: Farms closer to slaughterhouses experience higher vehicle
#   traffic (trucks transporting live animals) and greater fomite/spread
#   risk. Slaughterhouses are major nodes in the animal movement network
#   and can amplify FMD spread via contaminated vehicles or returned
#   animals. Higher score = closer proximity = higher risk.
# Direction: HIGHER value = HIGHER risk (direct, no inversion needed).
# Source: slaughterhouse_risk
combined_df['comp_slaughterhouse_risk'] = combined_df['slaughterhouse_risk']

components = [
    'comp_spatial_density', 'comp_animal_pressure', 'comp_network_risk',
    'comp_road_exposure',         'comp_border',           'comp_wildlife_javali',
    'comp_biosecurity',           'comp_fmdv_survival_prob_overall',
    'comp_slaughterhouse_risk',
]
print("    Raw risk component statistics:")
print(combined_df[components].describe().round(4).T.to_string())


# ==============================================================================
# SECTION 4 — NORMALIZATION: MIN-MAX SCALING TO [0, 1]
# ==============================================================================
# Formula:
#   norm(x) = (x - x_min) / (x_max - x_min)
#
# This maps each component independently to [0, 1].
#
# Inversion rule:
#   For variables where SMALLER raw value = HIGHER risk, we apply
#       risk_norm = 1 - norm(x)
#   so that in every normalized column, 1 = maximum risk, 0 = minimum risk.
#   Affected columns: accessibility, border, biosecurity.

print("\n[4] Normalizing risk components to [0, 1] ...")

# --------------------------------------------------------------------------
# Helper: min-max normalization
# --------------------------------------------------------------------------
def minmax_norm(series):
    """
    Applies min-max normalization to a pandas Series.
    Returns values in [0, 1].
    Formula: (x - min) / (max - min)
    """
    x_min = series.min()
    x_max = series.max()
    if x_max == x_min:
        # Constant column — assign 0 (no variation, no discriminatory power)
        return pd.Series(0.0, index=series.index)
    return (series - x_min) / (x_max - x_min)


# Spatial density: higher = higher risk → direct
combined_df['spatial_density_norm'] = minmax_norm(combined_df['comp_spatial_density'])

# Animal pressure: higher = higher risk → direct
combined_df['animal_pressure_norm'] = minmax_norm(combined_df['comp_animal_pressure'])

# Network risk: higher = higher risk → direct
combined_df['network_risk_norm'] = minmax_norm(combined_df['comp_network_risk'])

# Road exposure: HIGHER combined score = higher risk → direct (no inversion)
combined_df['road_exposure_norm'] = minmax_norm(combined_df['comp_road_exposure'])

# Border: SMALLER distance = higher risk → INVERT
combined_df['border_risk_norm'] = 1.0 - minmax_norm(combined_df['comp_border'])

# Wildlife (javali) reservoir risk: higher = higher risk → direct
combined_df['wildlife_javali_norm'] = minmax_norm(combined_df['comp_wildlife_javali'])

# Biosecurity: HIGHER score = more protected = LOWER risk → INVERT
combined_df['biosecurity_risk_norm'] = 1.0 - minmax_norm(combined_df['comp_biosecurity'])

# Susceptibility: higher survival prob = higher risk → direct
combined_df['comp_fmdv_survival_prob_overall_norm'] = minmax_norm(combined_df['comp_fmdv_survival_prob_overall'])

# Slaughterhouse proximity risk: higher = higher risk → direct
combined_df['slaughterhouse_risk_norm'] = minmax_norm(combined_df['comp_slaughterhouse_risk'])


# Define the ordered list of normalized component columns — used consistently below
NORM_COLS = [
    'spatial_density_norm',
    'animal_pressure_norm',
    'network_risk_norm',
    'road_exposure_norm',
    'border_risk_norm',
    'wildlife_javali_norm',
    'biosecurity_risk_norm',
    'comp_fmdv_survival_prob_overall_norm',
    'slaughterhouse_risk_norm',
]

print("    checkando ando: normalized component statistics (all must be in [0, 1]):")
print(combined_df[NORM_COLS].describe().round(4).T.to_string())

# Verify bounds
for col in NORM_COLS:
    assert combined_df[col].min() >= 0.0, f"ERROR: {col} has values below 0"
    assert combined_df[col].max() <= 1.0, f"ERROR: {col} has values above 1"


# Summary table of inversion decisions
inversion_table = pd.DataFrame({
    'component_norm':        NORM_COLS,
    'raw_source':            [
        'n_neighbors_5km', 'animals_within_5km', 'network_risk_score',
        'road_exposure_risk_raw', 'dist_border_m', 'javali_risk_smooth_iter',
        'biosecurity_farm_score', 'fmd_survival_prob_overall',
        'slaughterhouse_risk',
    ],
    'direction':             ['direct', 'direct', 'direct',
                              'direct', 'inverted', 'direct',
                              'inverted', 'direct', 'direct'],
    'rationale': [
        'more farms nearby = higher risk',
        'more animals nearby = higher risk',
        'more connected = higher risk',
        'higher combined road exposure (flow × decay proximity) = higher risk',
        'closer border = higher risk',
        'higher wild boar risk = higher wildlife reservoir risk',
        'higher biosecurity = LOWER risk',
        'virus survives longer = higher risk',
        'closer slaughterhouse = higher vehicle traffic risk',
    ],
})
print("\n    Component direction table:")
print(inversion_table.to_string(index=False))


# ==============================================================================
# SECTION 5 — COMPOSITE RISK SCORES
# ==============================================================================

# ------------------------------------------------------------------------------
# Score 1: Equal-weight risk score
# ------------------------------------------------------------------------------
# Each of the 8 components receives equal weight = 1/8 = 0.125.
# This is the simplest possible aggregation and serves as a reference baseline
# for comparing against the epidemiologically informed score.
# ------------------------------------------------------------------------------
N_COMPONENTS  = len(NORM_COLS)
EQUAL_WEIGHT  = 1.0 / N_COMPONENTS
EQUAL_WEIGHTS = {col: EQUAL_WEIGHT for col in NORM_COLS}

combined_df['risk_score_equal'] = sum(
    combined_df[col] * EQUAL_WEIGHT for col in NORM_COLS
)

print(f"\n    Equal-weight per component: {EQUAL_WEIGHT:.5f}")
print(f"    Equal-weight score — mean: {combined_df['risk_score_equal'].mean():.4f}, "
      f"std: {combined_df['risk_score_equal'].std():.4f}, "
      f"range: [{combined_df['risk_score_equal'].min():.4f}, "
      f"{combined_df['risk_score_equal'].max():.4f}]")

# ------------------------------------------------------------------------------
# Score 2: Epidemiologically weighted risk score
# ------------------------------------------------------------------------------
# Weights are grounded in the FMD epidemiological literature and expert
# reasoning. Full rationale is provided below. Weights sum to exactly 1.0.
#
# Component                  Weight    Rationale
# ─────────────────────────────────────────────────────────────────────────────
# animal_pressure_norm        0.1900  Animal density drives direct contact
#                                     transmission — dominant pathway at local
#                                     scale. Reduced from 0.2162: prior weight
#                                     created disproportionate ranking influence
#                                     (sensitivity: only 74 % top-500 stability
#                                     at ±20 % perturbation).
#                                     (Keeling & Rohani 2008; Kao 2002)
#
# network_risk_norm           0.1748  Live-animal movement is the principal
#                                     long-distance spread mechanism for FMD.
#                                     High leverage even if rare: a single
#                                     high-connectivity farm can seed outbreaks
#                                     across the state (Bigras-Poulin et al.
#                                     2006; Buhnerkempe et al. 2014).
#
# biosecurity_risk_norm       0.1300  Biosecurity measures directly reduce
#                                     per-contact transmission probability
#                                     (Nöremark et al. 2016). Weight reduced
#                                     from 0.1730: score variance is near-zero
#                                     (std = 0.018) in the Epicollect dataset,
#                                     so a higher weight provides no additional
#                                     discriminating power across farms.
#
# comp_fmdv_survival_prob_overall_norm
#                             0.1200  Environmental suitability for FMDV
#                                     survival (temperature-driven biphasic
#                                     decay). Increased from 0.0865: good
#                                     observed variance (std = 0.167) across
#                                     RS municipalities; modulates both entry
#                                     and persistence risk (Sanson 1994).
#
# border_risk_norm            0.1000  Proximity to Argentina/Uruguay is the
#                                     primary exotic-strain re-introduction
#                                     pathway for RS — the only Brazilian state
#                                     bordering both countries (Lyons et al.
#                                     2015). Increased from 0.0865 to reflect
#                                     the unique geopolitical exposure of RS.
#
# road_exposure_norm          0.0900  Combined road exposure (animal flow ×
#                                     exponential distance decay). Increased
#                                     from 0.0696 to reflect the new composite
#                                     formulation, which carries more information
#                                     than the previous simple distance metric
#                                     (Nigsch et al. 2013; Garner & Lack 1995).
#
# wildlife_javali_norm        0.0752  Wild boar (javali) reservoir risk —
#                                     municipality-level smoothed index.
#                                     Highest observed score variance (std = 0.338)
#                                     across components; weight held fixed to
#                                     preserve the epidemiological hierarchy
#                                     while respecting the coarser spatial
#                                     resolution of this input.
#
# spatial_density_norm        0.0700  Farm count within 5 km. Increased from
#                                     0.0602; partially correlated with
#                                     animal_pressure but captures shared
#                                     infrastructure and equipment risk not
#                                     reflected in head counts.
#
# slaughterhouse_risk_norm    0.0500  Slaughterhouse proximity. Reduced from
#                                     0.0600: overlap with road_exposure and
#                                     network components; 5 km threshold
#                                     limits applicability to a small farm
#                                     subset (11.9 % of properties).
#
# TOTAL                       1.00
# ─────────────────────────────────────────────────────────────────────────────
EPI_WEIGHTS = {
    'animal_pressure_norm':                  0.1900,
    'network_risk_norm':                     0.1748,
    'biosecurity_risk_norm':                 0.1300,
    'comp_fmdv_survival_prob_overall_norm':  0.1200,
    'border_risk_norm':                      0.1000,
    'road_exposure_norm':                    0.0900,
    'wildlife_javali_norm':                  0.0752,
    'spatial_density_norm':                  0.0700,
    'slaughterhouse_risk_norm':              0.0500,
}

# Verify weights sum to 1.0 (important for interpretability)
weight_sum = sum(EPI_WEIGHTS.values())
assert abs(weight_sum - 1.0) < 1e-9, (
    f"Epidemiological weights do not sum to 1.0 — got {weight_sum:.6f}. "
    f"Fix the EPI_WEIGHTS dictionary before continuing."
)

print(f"\n    Epidemiological weights (sum = {weight_sum:.4f}):")
for comp, w in EPI_WEIGHTS.items():
    print(f"      {comp:<30s}: {w:.2f}  (epi)   vs  {EQUAL_WEIGHT:.3f}  (equal)")

combined_df['risk_score_epi'] = sum(
    combined_df[col] * EPI_WEIGHTS[col] for col in NORM_COLS
)

print(f"\n    Epi-weight score  — mean: {combined_df['risk_score_epi'].mean():.4f}, "
      f"std: {combined_df['risk_score_epi'].std():.4f}, "
      f"range: [{combined_df['risk_score_epi'].min():.4f}, "
      f"{combined_df['risk_score_epi'].max():.4f}]")

# Correlation between the two scoring schemes
r_equal_epi = combined_df[['risk_score_equal', 'risk_score_epi']].corr(
    method='spearman'
).iloc[0, 1]
print(f"\n    Spearman r(equal, epi) = {r_equal_epi:.4f}")


# ==============================================================================
# SECTION 6 — SENSITIVITY ANALYSIS
# ==============================================================================
# Purpose:
#   Assess how much the top-farm ranking changes when each epidemiological
#   weight is nudged ±20 % relative to its base value.
#
# Method:
#   For each of the 8 components, two perturbed weight dictionaries are
#   created (one +20 %, one −20 %). The adjusted weight is renormalised so
#   the total always equals 1.0. The resulting 16 perturbed scores are
#   compared to the base epi-weighted score via:
#     (a) Spearman rank correlation (whole-population metric)
#     (b) Top-500 overlap percentage (priority-list stability metric)
#
#   If correlations remain > 0.99 and overlap > 95 %, the score is
#   considered robust to expert disagreement on weights.

print("\n[6] Running sensitivity analysis (±20 % weight perturbation) ...")

PERTURBATION = 0.20     # relative perturbation magnitude
TOP_N_FARMS  = 500      # number of top farms to track for rank stability

# Collect all score variants in a dict: label → pd.Series
sensitivity_scores = {
    'risk_score_equal': combined_df['risk_score_equal'],
    'risk_score_epi':   combined_df['risk_score_epi'],
}

for perturbed_col in NORM_COLS:
    for direction_label, sign in [('plus20', +1), ('minus20', -1)]:

        # Step 1: compute the new weight for the perturbed component
        base_w   = EPI_WEIGHTS[perturbed_col]
        delta    = base_w * PERTURBATION * sign
        new_w    = base_w + delta

        # Step 2: distribute the remaining budget proportionally
        #         among all other components
        remaining        = 1.0 - new_w
        other_total      = sum(v for k, v in EPI_WEIGHTS.items()
                               if k != perturbed_col)

        perturbed_weights = {}
        perturbed_weights[perturbed_col] = new_w
        for k, v in EPI_WEIGHTS.items():
            if k != perturbed_col:
                perturbed_weights[k] = v * (remaining / other_total)

        # Step 3: compute perturbed score
        short_label = (
            perturbed_col
            .replace('_norm', '')
            .replace('_risk', '')[:14]
        )
        label = f"{short_label}_{direction_label}"
        sensitivity_scores[label] = sum(
            combined_df[col] * perturbed_weights[col] for col in NORM_COLS
        )

# Build a DataFrame of all score variants for correlation computation
score_df    = pd.DataFrame(sensitivity_scores)
corr_matrix = score_df.corr(method='spearman')

print("\n    Spearman correlations (base scores shown):")
print(corr_matrix.loc[
    ['risk_score_equal', 'risk_score_epi'],
    ['risk_score_equal', 'risk_score_epi']
].round(4).to_string())

# Top-N rank stability vs the epi-weighted base score
base_top_idx = set(combined_df['risk_score_epi'].nlargest(TOP_N_FARMS).index)

stability_rows = []
for label, series in sensitivity_scores.items():
    if label in ('risk_score_equal', 'risk_score_epi'):
        continue
    perturbed_top_idx = set(series.nlargest(TOP_N_FARMS).index)
    overlap_pct       = len(base_top_idx & perturbed_top_idx) / TOP_N_FARMS * 100
    spearman_r        = corr_matrix.loc['risk_score_epi', label]
    stability_rows.append({
        'perturbed_variant':    label,
        'spearman_r_vs_epi':    round(spearman_r, 5),
        f'top{TOP_N_FARMS}_overlap_pct': round(overlap_pct, 2),
    })

stability_df = pd.DataFrame(stability_rows).sort_values(
    f'top{TOP_N_FARMS}_overlap_pct'
)

print(f"\n    Top-{TOP_N_FARMS} rank stability (sorted by worst overlap):")
print(stability_df.to_string(index=False))

# Min and max overlap across all perturbed variants
min_overlap = stability_df[f'top{TOP_N_FARMS}_overlap_pct'].min()
max_overlap = stability_df[f'top{TOP_N_FARMS}_overlap_pct'].max()
min_corr    = stability_df['spearman_r_vs_epi'].min()
print(f"\n    Overlap range : {min_overlap:.1f} % – {max_overlap:.1f} %")
print(f"    Min Spearman r: {min_corr:.5f}")



# ==============================================================================
# SECTION 7 — FINAL DATAFRAME
# ==============================================================================
# The final dataframe contains ALL farms in the combined dataset
# (already filtered to FMD-susceptible species upstream) with:
#   - All analysis columns (subset of the original dataset)
#   - All raw risk components (comp_*)
#   - All normalized components (*_norm)
#   - risk_score_equal
#   - risk_score_epi
# combined_df already contains everything — just take a copy for safety
final_df = combined_df.copy()


# --------------------------------------------------------------------------
# 8a. Risk score distributions
# --------------------------------------------------------------------------
for score_col in ['risk_score_equal', 'risk_score_epi']:
    d = final_df[score_col].describe()
    print(f"\n    {score_col}:")
    print(f"      n      = {d['count']:,.0f}")
    print(f"      mean   = {d['mean']:.4f}   std = {d['std']:.4f}")
    print(f"      min    = {d['min']:.4f}   max = {d['max']:.4f}")
    print(f"      Q1     = {d['25%']:.4f}   med = {d['50%']:.4f}   Q3 = {d['75%']:.4f}")

# Risk quartile labels (for summary table)
final_df['risk_quartile_epi'] = pd.qcut(
    final_df['risk_score_epi'], q=4,
    labels=['Q1_low', 'Q2_med_low', 'Q3_med_high', 'Q4_high']
)
print("\n    Farm counts per risk quartile (epi score):")
print(final_df['risk_quartile_epi'].value_counts().sort_index().to_string())

# --------------------------------------------------------------------------
# 8b. Top 50 high-risk farms by epi-weighted score
# --------------------------------------------------------------------------
top_50_epi = (
    final_df[
        ['COD_PROPRIEDADE_RURAL', 'risk_score_epi', 'risk_score_equal']
        + NORM_COLS
        + ['in_sisbravet']
    ]
    .sort_values('risk_score_epi', ascending=False)
    .head(50)
    .reset_index(drop=True)
)

print("\n    Top 10 farms by epi-weighted risk score:")
print(top_50_epi[['COD_PROPRIEDADE_RURAL', 'risk_score_epi',
                   'risk_score_equal', 'in_sisbravet']].head(10).to_string())


# Pairwise Spearman correlation among all normalized components
component_corr = final_df[NORM_COLS].corr(method='spearman')
print("\n    Spearman correlation matrix among normalized components:")
print(component_corr.round(3).to_string())


# ==============================================================================
# SECTION 9 — SAVE OUTPUTS
# ==============================================================================

# Drop the quartile label before saving (easy to recompute; avoids Categorical dtype)
final_df_to_save = final_df.drop(columns=['risk_quartile_epi'])

csv_path = str(OUTPUT_DIR / 'fmd_risk_scored.csv')
final_df_to_save.to_csv(csv_path, index=False)
print(f"     Saved CSV    : {csv_path}  ({len(final_df_to_save):,} rows × "
      f"{final_df_to_save.shape[1]} cols)")

# List all files written to the output folder
print("\n    Files in output directory:")
for f in sorted(os.listdir(str(OUTPUT_DIR))):
    fpath = os.path.join(str(OUTPUT_DIR), f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"      {f:<50s}  {size_kb:8.1f} KB")


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print(f"""
{'='*70}
  FMD RISK ANALYSIS COMPLETE
{'='*70}
  Input dataset       : {BANCO_FINAL_PKL.name}
  Farms processed     : {len(final_df_to_save):,}
  Columns in output   : {final_df_to_save.shape[1]}

  EQUAL-WEIGHT SCORE
    Mean   : {final_df['risk_score_equal'].mean():.4f}
    Std    : {final_df['risk_score_equal'].std():.4f}
    Range  : [{final_df['risk_score_equal'].min():.4f}, {final_df['risk_score_equal'].max():.4f}]

  EPI-WEIGHTED SCORE
    Mean   : {final_df['risk_score_epi'].mean():.4f}
    Std    : {final_df['risk_score_epi'].std():.4f}
    Range  : [{final_df['risk_score_epi'].min():.4f}, {final_df['risk_score_epi'].max():.4f}]

  Spearman r (equal vs epi)  : {r_equal_epi:.4f}
  Min top-{TOP_N_FARMS} overlap (sensitivity): {min_overlap:.1f} %

  Outputs saved to    : {OUTPUT_DIR}
{'='*70}
""")

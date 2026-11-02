# ==============================================================================
# FMD FARM-LEVEL RISK ANALYSIS
# ==============================================================================
# Purpose:
#   Produce two interpretable, composite FMD risk scores for every farm
#   in the state of Rio Grande do Sul (RS), Brazil. Note: the input
#   dataset is already filtered to FMD-susceptible species in
#   7_combine_and_check_data.py (Bovinos, Suíno, Caprinos, Bubalinos).
#       1. Equal-weight risk score  — all components weighted 1/8
#       2. Epidemiologically weighted risk score — literature-grounded weights
#
# Input:
#   intermediate/banco_final_to_risk_analysis.pkl
#
# Outputs (all saved to results/):
#   fmd_risk_scored.csv / .pkl      — full scored dataset (primary deliverable)
#   top50_high_risk_farms.csv       — top 50 farms by epi-weighted score
#   component_weights.csv           — weight table for reproducibility
#   risk_score_summary_stats.csv    — summary statistics
#   sensitivity_analysis.csv        — rank stability per perturbed weight
#   sensitivity_spearman_corr.csv   — full Spearman correlation matrix
#   plot1_risk_score_histograms.png
#   plot2_equal_vs_epi_scores.png
#   plot3_components_vs_epi_score.png
#   plot5_sensitivity_heatmap.png
#   plot7_component_correlation.png
#
# Key assumptions (documented inline at each step):
#   - network_risk_score NaN → median imputation (N printed at runtime)
#   - fmd_survival_prob_overall: HIGHER = more risk
#         (FMD virus environmental survival probability, temperature-driven)
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
# Author : Nicolas Cardenas
# Date   : 2026-04-03
# ==============================================================================

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')            # non-interactive backend — safe for saving
import matplotlib.pyplot as plt
import seaborn as sns
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
    "dominant_species",
    "QTD_DECLARADA_total",
    "network_risk_score",
    "n_neighbors_5km",
    "animal_pressure_weighted_10km",
    "road_exposure_risk_raw",
    "fmd_survival_prob_overall",
    "dist_border_m",
    "nearest_border_type",
    "javali_risk_smooth_iter",
    "slaughterhouse_risk"
]
missing_cols = [c for c in required_cols if c not in combined_df.columns]
assert not missing_cols, f"Missing required columns: {missing_cols}"

print("\n    FMD susceptible species (dominant per farm) in dataset:")
print(combined_df["dominant_species"].value_counts().to_string())

# Keep only analysis-relevant columns (avoid leaking IDs like CPF_CNPJ_TITULAR)
analysis_cols = [
    "COD_PROPRIEDADE_RURAL",
    "AREA_PROPRIEDADE",
    "dominant_species",
    "QTD_DECLARADA_total",
    "network_risk_score",
    "n_neighbors_5km",
    "animal_pressure_weighted_10km",
    "road_exposure_risk_raw",
    "fmd_survival_prob_overall",
    "dist_border_m",
    "nearest_border_type",
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
# --------------------------------------------------------------------------
cols_to_impute = [
    'n_neighbors_5km',
    'animal_pressure_weighted_10km',
    'road_exposure_risk_raw',
    'fmd_survival_prob_overall',
    'dist_border_m',
    'AREA_PROPRIEDADE',
    'QTD_DECLARADA_total',
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


# --- Component 1: Spatial density risk ---
combined_df['comp_spatial_density'] = combined_df['n_neighbors_5km']

# --- Component 2: Animal pressure risk ---
# Species-weighted sum of FMD-susceptible headcounts within 10 km true
# geodesic distance (swine weighted 50x cattle/sheep/goat/buffalo — see
# SPECIES_WEIGHTS in pipeline/3_get_spatial_net_features.py). More (weighted)
# animals = larger effective susceptible host pool = higher transmission
# pressure (Keeling & Rohani, 2008 — mass-action transmission assumption).
combined_df['comp_animal_pressure'] = combined_df['animal_pressure_weighted_10km']

# --- Component 3: Network / movement risk ---
combined_df['comp_network_risk'] = combined_df['network_risk_score']

# --- Component 4: Road exposure risk ---
combined_df['comp_road_exposure'] = combined_df['road_exposure_risk_raw']

# --- Component 5: Border risk ---
combined_df['comp_border'] = combined_df['dist_border_m']

# --- Component 6: Wildlife (javali) reservoir risk ---
combined_df['comp_wildlife_javali'] = combined_df['javali_risk_smooth_iter']

# --- Component 7: Epidemiological susceptibility ---
combined_df['comp_fmdv_survival_prob_overall'] = combined_df['fmd_survival_prob_overall']

# --- Component 8: Slaughterhouse proximity risk ---
combined_df['comp_slaughterhouse_risk'] = combined_df['slaughterhouse_risk']

components = [
    'comp_spatial_density', 'comp_animal_pressure', 'comp_network_risk',
    'comp_road_exposure',         'comp_border',           'comp_wildlife_javali',
    'comp_fmdv_survival_prob_overall',
    'comp_slaughterhouse_risk',
]
print("    Raw risk component statistics:")
print(combined_df[components].describe().round(4).T.to_string())


# ==============================================================================
# SECTION 4 — NORMALIZATION: MIN-MAX SCALING TO [0, 1]
# ==============================================================================

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
        'n_neighbors_5km', 'animal_pressure_weighted_10km', 'network_risk_score',
        'road_exposure_risk_raw', 'dist_border_m', 'javali_risk_smooth_iter',
        'fmd_survival_prob_overall',
        'slaughterhouse_risk',
    ],
    'direction':             ['direct', 'direct', 'direct',
                              'direct', 'inverted', 'direct',
                              'direct', 'direct'],
    'rationale': [
        'more farms nearby = higher risk',
        'more animals nearby = higher risk',
        'more connected = higher risk',
        'higher combined road exposure (flow × decay proximity) = higher risk',
        'closer border = higher risk',
        'higher wild boar risk = higher wildlife reservoir risk',
        'virus survives longer = higher risk',
        'closer slaughterhouse = higher vehicle traffic risk',
    ],
})
print("\n    Component direction table:")
print(inversion_table.to_string(index=False))

# Human-friendly labels for plots and tables
NICE_LABELS = {
    'spatial_density_norm': 'Spatial Density (norm.)',
    'animal_pressure_norm': 'Animal Pressure (norm.)',
    'network_risk_norm': 'Network / Movement Risk (norm.)',
    'road_exposure_norm': 'Road Exposure Risk (norm.)',
    'border_risk_norm': 'Border Proximity Risk (norm., inverted)',
    'wildlife_javali_norm': 'Wild Boar (Javali) Risk (norm.)',
    'comp_fmdv_survival_prob_overall_norm': 'FMDV Survival Suitability (norm.)',
    'slaughterhouse_risk_norm':             'Slaughterhouse Proximity Risk (norm.)',
}


# ==============================================================================
# SECTION 5 — COMPOSITE RISK SCORES
# ==============================================================================

# ------------------------------------------------------------------------------
# Score 1: Equal-weight risk score
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
# reasoning. Full rationale is provided below.
#
# The biosecurity component (base weight 0.13) has been removed.
# The eight surviving literature weights below sum to 0.87; they are rescaled
# (divided by 0.87) so the epidemiological weights again sum to exactly 1.0,
# preserving the relative epidemiological hierarchy among the eight components.
#
# Component                  Weight    Rationale
# ─────────────────────────────────────────────────────────────────────────────
# animal_pressure_norm        0.1900  Animal density drives direct contact
#                                     transmission — dominant pathway at local
#                                     scale. Reduced from 0.2162: prior weight
#                                     created disproportionate ranking influence
#                                     (sensitivity: only 74 % top-1000 stability
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
# SUBTOTAL (8 components)     0.87  → renormalized to 1.00 below
# ─────────────────────────────────────────────────────────────────────────────
# Base literature weights WITHOUT biosecurity (sum = 0.87).
EPI_WEIGHTS_BASE = {
    'animal_pressure_norm':                  0.1900,
    'network_risk_norm':                     0.1748,
    'comp_fmdv_survival_prob_overall_norm':  0.1200,
    'border_risk_norm':                      0.1000,
    'road_exposure_norm':                    0.0900,
    'wildlife_javali_norm':                  0.0752,
    'spatial_density_norm':                  0.0700,
    'slaughterhouse_risk_norm':              0.0500,
}

# Renormalize the base literature weights (sum = 0.87) to sum to 1.0.
_base_sum   = sum(EPI_WEIGHTS_BASE.values())
EPI_WEIGHTS = {k: v / _base_sum for k, v in EPI_WEIGHTS_BASE.items()}

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

print("\n[6] Running sensitivity analysis (±20 % weight perturbation) ...")

PERTURBATION = 0.20     # relative perturbation magnitude
TOP_N_FARMS  = 1000     # number of top farms to track for rank stability

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

# Save sensitivity outputs
stability_df.to_csv(str(OUTPUT_DIR / 'sensitivity_analysis.csv'), index=False)
corr_matrix.round(5).to_csv(str(OUTPUT_DIR / 'sensitivity_spearman_corr.csv'))
print("    Sensitivity tables saved.")



# ==============================================================================
# SECTION 7 — FINAL DATAFRAME
# ==============================================================================
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
    ]
    .sort_values('risk_score_epi', ascending=False)
    .head(50)
    .reset_index(drop=True)
)

print("\n    Top 10 farms by epi-weighted risk score:")
print(top_50_epi[['COD_PROPRIEDADE_RURAL', 'risk_score_epi',
                   'risk_score_equal']].head(10).to_string())

top_50_epi.to_csv(str(OUTPUT_DIR / 'top50_high_risk_farms.csv'), index=False)

# Pairwise Spearman correlation among all normalized components
component_corr = final_df[NORM_COLS].corr(method='spearman')
print("\n    Spearman correlation matrix among normalized components:")
print(component_corr.round(3).to_string())


# ==============================================================================
# SECTION 9 — VISUALIZATIONS
# ==============================================================================
sns.set_theme(style='whitegrid', font_scale=1.1)

# --------------------------------------------------------------------------
# Plot 1: Side-by-side histograms of both risk scores
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, colour, title in [
    (axes[0], 'risk_score_equal', 'steelblue',
     'Equal-Weight FMD Risk Score\nDistribution'),
    (axes[1], 'risk_score_epi',   'darkorange',
     'Epidemiologically Weighted FMD Risk Score\nDistribution'),
]:
    ax.hist(final_df[col], bins=80, color=colour, edgecolor='none', alpha=0.85)
    ax.axvline(final_df[col].mean(),   color='red',    linestyle='--',
               linewidth=1.5, label=f"Mean = {final_df[col].mean():.3f}")
    ax.axvline(final_df[col].median(), color='black',  linestyle=':',
               linewidth=1.5, label=f"Median = {final_df[col].median():.3f}")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Risk Score (0 – 1)', fontsize=11)
    ax.set_ylabel('Number of Farms',    fontsize=11)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'plot1_risk_score_histograms.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    Plot 1 saved: risk score histograms")

# --------------------------------------------------------------------------
# Plot 2: Equal-weight vs epi-weight scatter
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

ax.scatter(
    final_df['risk_score_equal'], final_df['risk_score_epi'],
    c='steelblue', alpha=0.25, s=3, linewidths=0, rasterized=True,
)

# Reference diagonal (equal = epi)
lims = [
    min(final_df['risk_score_equal'].min(), final_df['risk_score_epi'].min()),
    max(final_df['risk_score_equal'].max(), final_df['risk_score_epi'].max()),
]
ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5, label='y = x (reference)')

ax.set_xlabel('Equal-Weight Risk Score', fontsize=12)
ax.set_ylabel('Epi-Weighted Risk Score', fontsize=12)
ax.set_title(
    f'Equal-Weight vs Epi-Weighted Risk Scores\nSpearman r = {r_equal_epi:.4f}',
    fontsize=12
)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'plot2_equal_vs_epi_scores.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    Plot 2 saved: equal vs epi score scatter")

# --------------------------------------------------------------------------
# Plot 3: Top-4 highest-weight components vs epi risk score
#         (subsampled to 10,000 points for readability)
# --------------------------------------------------------------------------
top_components = sorted(
    EPI_WEIGHTS.items(), key=lambda x: x[1], reverse=True
)[:4]
scatter_vars = [
    (col, f"{NICE_LABELS.get(col, col)} (w={w:.3f})")
    for col, w in top_components
]

rng = np.random.default_rng(42)
sample_idx = rng.choice(len(final_df), size=min(10_000, len(final_df)), replace=False)
sample     = final_df.iloc[sample_idx].copy()

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

for i, (col, label) in enumerate(scatter_vars):
    axes[i].scatter(
        sample[col], sample['risk_score_epi'],
        c='steelblue', alpha=0.35, s=5, linewidths=0,
    )
    m, b = np.polyfit(sample[col], sample['risk_score_epi'], 1)
    x_line = np.linspace(sample[col].min(), sample[col].max(), 200)
    axes[i].plot(x_line, m * x_line + b, color='black',
                 linewidth=1.5, linestyle='-', alpha=0.7,
                 label=f'Linear fit (r²={np.corrcoef(sample[col], sample["risk_score_epi"])[0,1]**2:.3f})')
    axes[i].set_xlabel(label,                   fontsize=10)
    axes[i].set_ylabel('Epi-Weighted Risk Score', fontsize=10)
    axes[i].set_title(f'{label}\nvs Epi Risk Score',   fontsize=10)
    axes[i].legend(fontsize=8)

plt.suptitle(
    'Key Risk Components vs Epi-Weighted Score\n(10,000 farm random sample)',
    fontsize=13, y=1.01
)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'plot3_components_vs_epi_score.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    Plot 3 saved: component scatter plots")

# --------------------------------------------------------------------------
# Plot 5: Sensitivity analysis — Spearman correlation heatmap
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 13))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.3f',
    cmap='RdYlGn',
    vmin=0.0,
    vmax=1.0,
    ax=ax,
    linewidths=0.4,
    annot_kws={'size': 7},
    square=True,
    cbar_kws={'label': 'Spearman correlation'}
)

ax.set_title(
    'Spearman Correlations: Base vs ±20 % Perturbed Risk Scores\n'
    '(Sensitivity Analysis - higher = more robust to weight changes)',
    fontsize=12
)
ax.set_xlabel('Scenario')
ax.set_ylabel('Scenario')

plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'plot5_sensitivity_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    Plot 5 saved: sensitivity heatmap")

# --------------------------------------------------------------------------
# Plot 7: Spearman correlation matrix of normalized components
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(
    component_corr,
    annot=True, fmt='.2f',
    cmap='coolwarm', center=0,
    vmin=-1, vmax=1,
    ax=ax, linewidths=0.5,
    annot_kws={'size': 14},
    square=True,
)
tick_labels = [NICE_LABELS.get(c, c) for c in component_corr.columns]
ax.set_xticklabels(tick_labels, rotation=40, ha='right', fontsize=9)
ax.set_yticklabels(tick_labels, rotation=0,             fontsize=9)
ax.set_title('Spearman Correlation Among Normalised Risk Components', fontsize=16)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'plot7_component_correlation.png'), dpi=300, bbox_inches='tight')
plt.close()
print("    Plot 7 saved: component correlation heatmap")


# ==============================================================================
# SECTION 10 — SAVE OUTPUTS
# ==============================================================================

# Drop the quartile label before saving (easy to recompute; avoids Categorical dtype)
final_df_to_save = final_df.drop(columns=['risk_quartile_epi'])

# 10a. Full scored dataframe — pickle
pkl_path = str(OUTPUT_DIR / 'fmd_risk_scored.pkl')
final_df_to_save.to_pickle(pkl_path)
print(f"     Saved pickle : {pkl_path}  ({len(final_df_to_save):,} rows × "
      f"{final_df_to_save.shape[1]} cols)")

# 10b. Full scored dataframe — CSV
csv_path = str(OUTPUT_DIR / 'fmd_risk_scored.csv')
final_df_to_save.to_csv(csv_path, index=False)
print(f"     Saved CSV    : {csv_path}")

# 10c. Summary statistics table
summary_stats = (
    final_df_to_save[['risk_score_equal', 'risk_score_epi'] + NORM_COLS]
    .describe().T.round(4)
)
summary_stats.to_csv(str(OUTPUT_DIR / 'risk_score_summary_stats.csv'))
print("     Saved: risk_score_summary_stats.csv")

# 10d. Component weight table (for full reproducibility)
weight_table = pd.DataFrame([
    {
        'component_norm':     col,
        'raw_source':         inversion_table.set_index('component_norm').loc[col, 'raw_source'],
        'direction':          inversion_table.set_index('component_norm').loc[col, 'direction'],
        'equal_weight':       round(EQUAL_WEIGHT, 5),
        'epi_weight':         EPI_WEIGHTS[col],
        'epi_vs_equal_ratio': round(EPI_WEIGHTS[col] / EQUAL_WEIGHT, 3),
    }
    for col in NORM_COLS
])
weight_table.to_csv(str(OUTPUT_DIR / 'component_weights.csv'), index=False)
print("     Saved: component_weights.csv")

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

#!/usr/bin/env python3
"""
CADASTRO RISK SCORING - SINGLE MODEL + MEAN FALLBACK
================================================================================
Flat step-by-step script (no functions). All logic inlined in order.

Strategy:
  - Train multiple models on Epicollect biosecurity features (visited farms only)
  - Select best model using CV ROC AUC
  - Score visited farms with best model
  - Assign population mean score to unvisited farms
  - Rank all 270,478 farms

Outputs:
  intermediate/cadastro_with_risk.pkl / .csv
  intermediate/r_plots/
    01_feature_importance.csv
    02_roc_curves_all_folds.csv
    03_risk_score_by_sisbravet.csv
    04_model_comparison.csv
    05_main_results.csv
"""

import sys
import pickle
import warnings
import unicodedata
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import (
    SISBRAVET_PKL, CADASTRO_PKL, EPICOLLECT_CSV,
    INTERMEDIATE, R_PLOTS_DIR, BIOSEC_PKL,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR  = INTERMEDIATE
OUTPUT_FILE = "cadastro_with_risk.pkl"

ID_COLUMN     = "COD_PROPRIEDADE_RURAL"
TARGET_COLUMN = "in_fmd_case"
RANDOM_SEED   = 42
TEST_SIZE     = 0.2

EPICOLLECT_FEATURES = [
    '13_6_A_propriedade_t', '14_7_Havia_placa_de_', '15_8_Na_entrada_da_p',
    '17_Nmero_de_Bovinos', '18_Nmero_de_Bubalino', '19_Nmero_de_Sunos',
    '20_Nmero_de_Caprinos', '21_Nmero_de_Ovinos', '22_9_H_diferentes_es',
    '23_10_Finalidade_Pri', '24_11_Qual_o_destino', '25_12_Tipo_de_criao_',
    '26_13_Grau_de_tecnif', '27_14_H_animais_conf', '28_15_O_pastopiquete',
    '29_16_H_fonte_de_gua', '30_17_A_propriedade_', '31_18_Com_que_frequn',
    '32_19_H_compartilham', '33_20_Com_que_frequn', '34_21_Aps_retornar_d',
    '35_22_Com_que_frequn', '36_23_Os_animais_rec', '37_24_Com_que_frequn',
    '38_25_Qual_o_mtodo_d', '39_26_H_fornecimento', '40_27_H_compartilham',
    '41_28_H_compartilham', '42_29_H_entrada_de_v', '43_30_H_compartilham',
    '44_31_A_propriedade_', '46_32_Ao_fazer_a_ins', '47_33_Ao_fazer_inspe',
    '48_34_Durante_a_sua_', '50_35_Com_qual_frequ', '51_36_O_Sr_a_conhece',
    '52_37_O_Sr_a_sabe_co', '54_38_Senhora_veteri', '55_Bovinos',
    '56_Bubalinos', '57_Sunos', '58_Caprinos', '59_Ovinos'
]

print("\n" + "=" * 70)
print("CADASTRO RISK SCORING - SINGLE MODEL + MEAN FALLBACK")
print("=" * 70)

# ============================================================================
# STEP 1: Load data
# ============================================================================
with open(str(SISBRAVET_PKL), 'rb') as f:
    sisbravet = pickle.load(f)

with open(str(CADASTRO_PKL), 'rb') as f:
    cadastro = pickle.load(f)

epicollect = pd.read_csv(str(EPICOLLECT_CSV))
epicollect = epicollect.rename(columns={'4_3_Cdigo_do_Estabel': ID_COLUMN})
epicollect[ID_COLUMN] = epicollect[ID_COLUMN].astype(str)

# ============================================================================
# STEP 2: Label cadastro
# Match CPF_CNPJ_TITULAR with sisbravet CPF, constrained to municipality.
# sisbravet only has municipality name; cadastro only has COD_MUNICIPIO_IBGE.
# Bridge: load RS municipality map via geobr and join on normalized name.
# ============================================================================
import geobr  # noqa: E402  (local import keeps geobr optional for other steps)

def _normalize_muni(s):
    """Uppercase, strip, and remove diacritics from a municipality name."""
    if pd.isna(s):
        return ''
    s = str(s).upper().strip()
    s = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in s if not unicodedata.combining(c))

print("Loading RS municipality map (geobr)...")
rs_munis = geobr.read_municipality(code_muni='RS', year=2020)
rs_munis['_name_norm'] = rs_munis['name_muni'].apply(_normalize_muni)
rs_munis['_ibge'] = rs_munis['code_muni'].astype(int)
muni_lookup = rs_munis.set_index('_name_norm')['_ibge'].to_dict()  # name → IBGE code

cadastro_labeled = cadastro.copy()
cadastro_labeled[ID_COLUMN] = cadastro_labeled[ID_COLUMN].astype(str)
cadastro_labeled[TARGET_COLUMN] = False

if 'CPF' in sisbravet.columns and 'CPF_CNPJ_TITULAR' in cadastro_labeled.columns:
    # Prepare sisbravet: numeric CPF + IBGE code resolved from municipality name
    sisbravet_cpf = sisbravet[sisbravet['CPF'].notna()].copy()
    sisbravet_cpf['_cpf'] = pd.to_numeric(sisbravet_cpf['CPF'], errors='coerce')
    sisbravet_cpf['_muni_norm'] = sisbravet_cpf['Município'].apply(_normalize_muni)
    sisbravet_cpf['_ibge'] = sisbravet_cpf['_muni_norm'].map(muni_lookup)

    unresolved = sisbravet_cpf[sisbravet_cpf['_ibge'].isna()]['_muni_norm'].unique()
    if len(unresolved):
        print(f"  WARNING: {len(unresolved)} sisbravet municipality name(s) not matched "
              f"to IBGE code: {unresolved}")

    # Unique (CPF, IBGE) pairs from confirmed FMD occurrences
    fmd_pairs = (
        sisbravet_cpf[['_cpf', '_ibge']]
        .dropna()
        .drop_duplicates()
    )
    fmd_pairs['_ibge'] = fmd_pairs['_ibge'].astype(int)

    # Match cadastro on both CPF and municipality
    cadastro_labeled['_cpf'] = pd.to_numeric(cadastro_labeled['CPF_CNPJ_TITULAR'], errors='coerce')
    matched_mask = (
        cadastro_labeled[['_cpf', 'COD_MUNICIPIO_IBGE']]
        .merge(
            fmd_pairs,
            left_on=['_cpf', 'COD_MUNICIPIO_IBGE'],
            right_on=['_cpf', '_ibge'],
            how='left',
            indicator=True,
        )['_merge']
        .eq('both')
        .values  # numpy bool array aligned with cadastro_labeled rows
    )
    cadastro_labeled.loc[matched_mask, TARGET_COLUMN] = True
    cadastro_labeled.drop(columns=['_cpf'], inplace=True)

    print(f"  FMD-labeled farms (CPF + municipality): {matched_mask.sum():,} "
          f"of {len(cadastro_labeled):,}")

# ============================================================================
# STEP 3: Merge epicollect
# ============================================================================
label_map = cadastro_labeled[[ID_COLUMN, TARGET_COLUMN]].drop_duplicates(ID_COLUMN)

epicollect_labeled = epicollect.merge(label_map, on=ID_COLUMN, how='left')
epicollect_labeled[TARGET_COLUMN] = epicollect_labeled[TARGET_COLUMN].fillna(0).astype(int)

# ============================================================================
# STEP 4: Features
# ============================================================================
feat_cols = [c for c in EPICOLLECT_FEATURES if c in epicollect_labeled.columns]

X = epicollect_labeled[feat_cols]
y = epicollect_labeled[TARGET_COLUMN].astype(int)

# ============================================================================
# STEP 5: Preprocessing
# ============================================================================
numeric_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(epicollect_labeled[c])]
categorical_cols = [c for c in feat_cols if c not in numeric_cols]

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, numeric_cols),
    ('cat', cat_pipe, categorical_cols)
])

# ============================================================================
# STEP 6: Models
# ============================================================================
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_SEED),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_SEED)
}

cv_results = {}
best_model_name = None
best_score = -1

# ============================================================================
# STEP 7: CV evaluation
# ============================================================================
for name, model in models.items():
    fold_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe = Pipeline([
            ('prep', clone(preprocessor)),
            ('clf', clone(model))
        ])

        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, prob)
        fold_scores.append(auc)

    cv_results[name] = {
        'mean': np.mean(fold_scores),
        'std': np.std(fold_scores),
        'folds': fold_scores
    }

    if np.mean(fold_scores) > best_score:
        best_score = np.mean(fold_scores)
        best_model_name = name

# ============================================================================
# STEP 8: ROC curves for ALL models (FIXED)
# ============================================================================
roc_rows = []

print("Collecting ROC curves for all models")

for model_name, model in models.items():
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y)):

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe = Pipeline([
            ('prep', clone(preprocessor)),
            ('clf', clone(model))
        ])

        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_val)[:, 1]

        fpr, tpr, _ = roc_curve(y_val, prob)
        auc = roc_auc_score(y_val, prob)

        for fp, tp in zip(fpr, tpr):
            roc_rows.append({
                'model': model_name,
                'fold': fold_i + 1,
                'fpr': fp,
                'tpr': tp,
                'fold_auc': auc
            })

# ============================================================================
# STEP 9: Fit best model
# ============================================================================
best_model = Pipeline([
    ('prep', clone(preprocessor)),
    ('clf', clone(models[best_model_name]))
])

best_model.fit(X, y)

# ============================================================================
# STEP 10: Score farms
# ============================================================================
epicollect_labeled['risk_score'] = best_model.predict_proba(X)[:, 1]

visited = epicollect_labeled.groupby(ID_COLUMN)['risk_score'].mean().reset_index()
pop_mean = visited['risk_score'].mean()

# ============================================================================
# STEP 11: Apply to cadastro
# ============================================================================
scores = cadastro_labeled.merge(visited, on=ID_COLUMN, how='left')
scores['score_source'] = np.where(scores['risk_score'].notna(), 'model', 'mean')
scores['risk_score'] = scores['risk_score'].fillna(pop_mean)

# ============================================================================
# STEP 12: Export ROC file (ALL MODELS INCLUDED)
# ============================================================================
R_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
roc_df = pd.DataFrame(roc_rows)
roc_df.to_csv(str(R_PLOTS_DIR / "02_roc_curves_all_folds.csv"), index=False)

# ============================================================================
# STEP 13: Model comparison
# ============================================================================
model_comp = []
for m, r in cv_results.items():
    for i, v in enumerate(r['folds']):
        model_comp.append({
            'model': m,
            'fold': i + 1,
            'auc': v,
            'mean': r['mean']
        })

pd.DataFrame(model_comp).to_csv(str(R_PLOTS_DIR / "04_model_comparison.csv"), index=False)

# Save the scored dataset
scores.to_pickle(str(BIOSEC_PKL))
print(f"Saved: {BIOSEC_PKL}")
print("DONE")

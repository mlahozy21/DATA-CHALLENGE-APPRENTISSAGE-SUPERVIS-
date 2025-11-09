# --- Fichero: config.py ---

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

# --- Constantes Globales ---
RANDOM_STATE = 42
N_SPLITS = 5 # Nombre de plis pour le Stacking OOF

# --- Chemins des Données ---
# Modifie DATA_DIR pour qu'il pointe vers le bon dossier
# (ex: "../data/" si les données sont un niveau au-dessus)
DATA_DIR = "data/" 
TRAIN_FILE = DATA_DIR + "train_data2.csv"
TEST_FILE = DATA_DIR + "test_data2.csv"
SUBMISSION_FILE = DATA_DIR + "submission2.csv"

# --- Chemins des Modèles et Objets ---
# Répertoire pour sauvegarder les modèles
MODEL_DIR = "models_stacking/"

# Modèles de base (L0)
RF_MODEL_PATH = MODEL_DIR + "stack_rf_base.joblib"
LGBM_MODEL_PATH = MODEL_DIR + "stack_lgbm_base.joblib"
CAT_MODEL_PATH = MODEL_DIR + "stack_cat_base.joblib"

# Modèle final (L1)
META_MODEL_PATH = MODEL_DIR + "stack_meta_model.joblib"

# Processeurs et Encoders
PREPROCESSOR_A_PATH = MODEL_DIR + "stack_preprocessor_A_rf.joblib"
LE_Y_PATH = MODEL_DIR + "stack_label_encoder_y.joblib"

# Méta-features (Cache)
CACHE_DIR = "cache_stacking/"
META_TRAIN_PATH = CACHE_DIR + "meta_train.npy"
META_TEST_PATH = CACHE_DIR + "meta_test.npy"
Y_FULL_PATH = CACHE_DIR + "y_full.npy"
TEST_IDS_PATH = CACHE_DIR + "test_ids.joblib"


# --- Définition des Paramètres des Modèles ---

# 1. Paramètres RF (pour Pipeline)
rf_params = {
    'n_estimators': 400, 
    'random_state': RANDOM_STATE, 
    'class_weight': 'balanced', 
    'n_jobs': -1
}

# 2. Paramètres LGBM (King)
lgbm_king_params = {
    'n_estimators': 3000, 
    'learning_rate': 0.03, 
    'num_leaves': 35,
    'min_child_samples': 100, 
    'subsample': 0.8, 
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5, 
    'reg_lambda': 0.5,
    'class_weight': {0: 1.0, 1: 1.0, 2: 8.0},
    'random_state': RANDOM_STATE, 
    'n_jobs': -1
}
# Callbacks pour LGBM
lgbm_callbacks = [lgb.early_stopping(100, verbose=False)]

# 3. Paramètres CatBoost (Challenger)
cat_class_weights = [1.0, 1.0, 0.7]
catboost_challenger_params = {
    'loss_function': 'MultiClassOneVsAll', 
    'eval_metric': 'TotalF1:average=Macro',
    'iterations': 20000, 
    'learning_rate': 0.05, 
    'depth': 8, 
    'l2_leaf_reg': 2.0,
    'early_stopping_rounds': 1000, 
    'bootstrap_type': 'No', 
    'rsm': 1.0,
    'random_strength': 0.0, 
    'border_count': 254, 
    'auto_class_weights': None,
    'class_weights': cat_class_weights, 
    'one_hot_max_size': 50,
    'max_ctr_complexity': 4, 
    'random_state': RANDOM_STATE, 
    'verbose': False,
}

# 4. Paramètres Gestionnaire L1 (Meta-Modèle)
final_estimator_params = {
    'max_iter': 1000, 
    'random_state': RANDOM_STATE
}
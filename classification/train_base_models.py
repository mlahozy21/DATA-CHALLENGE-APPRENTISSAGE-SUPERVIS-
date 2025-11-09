# --- Fichero: train_base_models.py ---

import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

# Importer nos modules personnalisés
import c.config as config
import c.data_loader as dl

# Ignorer les avertissements
warnings.filterwarnings('ignore')

def train_l0_experts():
    """
    Entraîne les 3 experts de Niveau 0 (RF, LGBM, CatBoost) en utilisant
    une stratégie Out-of-Fold (OOF) pour générer les méta-features L1.
    Sauvegarde ensuite les modèles L0 finaux et les méta-features.
    """
    
    # --- 1. Charger les 3 ensembles de données préparés ---
    print("--- Étape 1/4 : Chargement des 3 ensembles de données... ---")
    datasets, y_full = dl.load_and_preprocess_data()
    if datasets is None:
        print("Échec du chargement des données. Arrêt.")
        return

    # Dépaqueter les ensembles de données
    X_A_full, X_A_test = datasets['A_rf']
    X_B_full, X_B_test, cats_B = datasets['B_lgbm']
    X_C_full, X_C_test, cats_C = datasets['C_cat']
    
    # Charger le préprocesseur pour le Pipeline RF
    preprocessor_rf = joblib.load(config.PREPROCESSOR_A_PATH)
    n_classes = len(np.unique(y_full))

    # --- 2. Définir les experts L0 et les pipelines ---
    print("--- Étape 2/4 : Initialisation des experts L0... ---")
    
    # Expert 1: Pipeline RandomForest
    rf_model = RandomForestClassifier(**config.rf_params)
    rf_pipeline = make_pipeline(preprocessor_rf, rf_model)

    # Expert 2: LGBMClassifier
    lgbm_king = lgb.LGBMClassifier(**config.lgbm_king_params)

    # Expert 3: CatBoostClassifier
    catboost_challenger = CatBoostClassifier(**config.catboost_challenger_params)

    # --- 3. Entraînement OOF (Out-of-Fold) pour générer X_meta_train ---
    print(f"--- Étape 3/4 : Démarrage de l'entraînement OOF (K={config.N_SPLITS})... ---")
    
    kf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    # Initialiser les conteneurs pour les prédictions OOF (méta-features L1)
    meta_train_rf = np.zeros((len(X_A_full), n_classes))
    meta_train_lgbm = np.zeros((len(X_B_full), n_classes))
    meta_train_cat = np.zeros((len(X_C_full), n_classes))

    # Conteneurs pour les prédictions de test (pour la moyenne)
    meta_test_rf_folds = np.zeros((len(X_A_test), n_classes, config.N_SPLITS))
    meta_test_lgbm_folds = np.zeros((len(X_B_test), n_classes, config.N_SPLITS))
    meta_test_cat_folds = np.zeros((len(X_C_test), n_classes, config.N_SPLITS))

    # Listes pour stocker les meilleures itérations (pour LGBM et CatBoost)
    lgbm_best_iters = []
    catboost_best_iters = []

    # Boucle de validation croisée
    # Note: Nous utilisons les indices de X_B_full comme référence, 
    # car ils devraient tous avoir la même longueur (len(y_full))
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_B_full, y_full)):
        print(f"--- Fold (Pli) {fold + 1}/{config.N_SPLITS} ---")
        y_train_fold, y_val_fold = y_full[train_idx], y_full[val_idx]

        # --- Expert 1: RandomForest (Données A) ---
        print("  Entraînement RF...")
        X_A_train_fold, X_A_val_fold = X_A_full.iloc[train_idx], X_A_full.iloc[val_idx]
        rf_clone = clone(rf_pipeline)
        rf_clone.fit(X_A_train_fold, y_train_fold)
        meta_train_rf[val_idx] = rf_clone.predict_proba(X_A_val_fold)
        meta_test_rf_folds[:, :, fold] = rf_clone.predict_proba(X_A_test)

        # --- Expert 2: LGBM (Données B) ---
        print("  Entraînement LGBM...")
        X_B_train_fold, X_B_val_fold = X_B_full.iloc[train_idx], X_B_full.iloc[val_idx]
        lgbm_clone = clone(lgbm_king)
        lgbm_clone.fit(
            X_B_train_fold, y_train_fold,
            eval_set=[(X_B_val_fold, y_val_fold)],
            eval_metric='multi_logloss',
            callbacks=config.lgbm_callbacks,
            categorical_feature=cats_B
        )
        meta_train_lgbm[val_idx] = lgbm_clone.predict_proba(X_B_val_fold)
        meta_test_lgbm_folds[:, :, fold] = lgbm_clone.predict_proba(X_B_test)
        lgbm_best_iters.append(lgbm_clone.best_iteration_ if lgbm_clone.best_iteration_ else config.lgbm_king_params['n_estimators'])

        # --- Expert 3: CatBoost (Données C) ---
        print("  Entraînement CatBoost...")
        X_C_train_fold, X_C_val_fold = X_C_full.iloc[train_idx], X_C_full.iloc[val_idx]
        cat_clone = clone(catboost_challenger)
        cat_clone.fit(
            X_C_train_fold, y_train_fold,
            eval_set=(X_C_val_fold, y_val_fold),
            cat_features=cats_C
        )
        meta_train_cat[val_idx] = cat_clone.predict_proba(X_C_val_fold)
        meta_test_cat_folds[:, :, fold] = cat_clone.predict_proba(X_C_test)
# LÍNEA CORREGIDA:
        catboost_best_iters.append(cat_clone.get_best_iteration() if cat_clone.get_best_iteration() else config.catboost_challenger_params['iterations'])
    print("--- Entraînement OOF L0 terminé. ---")

    # --- 4. Finalisation et Sauvegarde ---
    print("--- Étape 4/4 : Finalisation et Sauvegarde des modèles L0 et des méta-features... ---")

    # Combiner les méta-features L1 pour l'entraînement
    X_meta_train = np.concatenate([meta_train_rf, meta_train_lgbm, meta_train_cat], axis=1)

    # Calculer la moyenne des prédictions de test des K-folds
    # C'est une méthode courante pour générer les méta-features de test
    meta_test_rf = np.mean(meta_test_rf_folds, axis=2)
    meta_test_lgbm = np.mean(meta_test_lgbm_folds, axis=2)
    meta_test_cat = np.mean(meta_test_cat_folds, axis=2)
    
    # Combiner les méta-features L1 pour le test
    X_meta_test = np.concatenate([meta_test_rf, meta_test_lgbm, meta_test_cat], axis=1)

    # Sauvegarder les méta-features (cache L1)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.save(config.META_TRAIN_PATH, X_meta_train)
    np.save(config.META_TEST_PATH, X_meta_test)
    np.save(config.Y_FULL_PATH, y_full)
    print(f"Méta-features (L1) sauvegardées dans '{config.CACHE_DIR}'")

    # --- Optionnel : Entraîner et sauvegarder les modèles L0 finaux sur 100% des données ---
    # C'est une alternative à la moyenne des K-folds, mais nous suivons la logique OOF
    # pour le test également. Cependant, sauvegardons UN modèle final
    # (entraîné avec le nombre moyen d'itérations) pour référence.
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # 1. RF Final
    print("  Sauvegarde du pipeline RF final (entraîné sur le Fold 1)...")
    # Note : Le clonage est déjà fait, nous pourrions sauvegarder le 'rf_clone' du dernier fold,
    # ou entraîner un nouveau modèle sur 100% des données.
    # Pour la simplicité, nous allons entraîner et sauvegarder de nouveaux modèles finaux.
    print("  Entraînement du modèle RF final sur 100% des données...")
    rf_pipeline_final = clone(rf_pipeline)
    rf_pipeline_final.fit(X_A_full, y_full)
    joblib.dump(rf_pipeline_final, config.RF_MODEL_PATH)

    # 2. LGBM Final
    avg_lgbm_iters = int(np.mean(lgbm_best_iters) * 1.1) # Un petit buffer
    print(f"  Entraînement du LGBM final ({avg_lgbm_iters} itérations)...")
    lgbm_king_final = clone(lgbm_king)
    lgbm_king_final.set_params(n_estimators=avg_lgbm_iters, early_stopping_round=None)
    lgbm_king_final.fit(X_B_full, y_full, categorical_feature=cats_B)
    joblib.dump(lgbm_king_final, config.LGBM_MODEL_PATH)
    
    # 3. CatBoost Final
    avg_cat_iters = int(np.mean(catboost_best_iters) * 1.1) # Un petit buffer
    print(f"  Entraînement du CatBoost final ({avg_cat_iters} itérations)...")
    catboost_final = clone(catboost_challenger)
    catboost_final.set_params(iterations=avg_cat_iters, early_stopping_rounds=None)
    catboost_final.fit(X_C_full, y_full, cat_features=cats_C)
    joblib.dump(catboost_final, config.CAT_MODEL_PATH)
    
    print(f"Modèles L0 finaux sauvegardés dans '{config.MODEL_DIR}'")
    print("--- Processus d'entraînement L0 terminé. ---")


if __name__ == "__main__":
    train_l0_experts()
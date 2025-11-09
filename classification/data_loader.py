# --- Fichero: data_loader.py ---

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib
import os

# Importer nos configurations et fonctions de features
import c.config as config
import c.feature_engineering as fe

def basic_clean(df, is_train=True):
    """Applique le nettoyage de base (suppression de colonnes, filtres)."""
    df_clean = df.drop(['Unnamed: 0', 'row_id'], axis=1, errors='ignore')
    
    if is_train:
        # Appliquer les filtres uniquement sur l'ensemble d'entraînement
        df_clean = df_clean[(df_clean['adults'] > 0) | (df_clean['children'] > 0) | (df_clean['babies'] > 0)]
        df_clean = df_clean[df_clean['adr'] < 5000].copy()
        
    return df_clean

def load_and_preprocess_data():
    """
    Charge les données brutes, nettoie, et crée les 3 pipelines de données (A, B, C)
    pour chaque expert (RF, LGBM, CatBoost).
    """
    print("--- Étape : Chargement et Nettoyage de Base ---")
    try:
        train_df_raw = pd.read_csv(config.TRAIN_FILE)
        test_df_raw = pd.read_csv(config.TEST_FILE)
    except FileNotFoundError as e:
        print(f"Erreur: Fichier de données non trouvé. {e}")
        return None

    # Sauvegarder les IDs de test pour la soumission finale
    test_ids = test_df_raw['row_id']
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    joblib.dump(test_ids, config.TEST_IDS_PATH)

    # Nettoyage de base
    train_df = basic_clean(train_df_raw, is_train=True)
    test_df_cleaned = basic_clean(test_df_raw, is_train=False)

    # Encodage de la variable cible (y)
    le_y = LabelEncoder()
    y_full = le_y.fit_transform(train_df['reservation_status'].values)
    
    # Sauvegarder l'encodeur pour la prédiction finale
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(le_y, config.LE_Y_PATH)
    
    X_full_base = train_df.drop('reservation_status', axis=1)
    
    print(f"Données chargées. {len(y_full)} échantillons d'entraînement.")
    
    # --- Création des 3 ensembles de données spécialisés ---

    # --- 3.1 Données A (P1): Pour RF (OHE + Scaler) ---
    print("--- Préparation Données A (RF)... ---")
    X_A_full_raw = X_full_base.copy()
    X_A_test_raw = test_df_cleaned.copy()
    
    # Aligner les colonnes
    X_A_full, X_A_test = X_A_full_raw.align(X_A_test_raw, join='inner', axis=1)
    
    numerical_cols_A = X_A_full.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_A = X_A_full.select_dtypes(include='object').columns.tolist()

    # Définir le préprocesseur
    preprocessor_ohe = ColumnTransformer(
        transformers=[
            ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numerical_cols_A),
            ('cat', make_pipeline(SimpleImputer(strategy='most_frequent'),
                                  OneHotEncoder(handle_unknown='ignore', sparse_output=False)), categorical_cols_A)
        ],
        remainder='passthrough'
    )
    # NOTE: Ce préprocesseur sera intégré dans un Pipeline avec le RF dans le script d'entraînement
    # Nous sauvegardons ce préprocesseur pour l'entraînement OOF
    joblib.dump(preprocessor_ohe, config.PREPROCESSOR_A_PATH)
    

    # --- 3.2 Données B (P2-King): Pour LGBM (FE Cyclique + Label Encoding) ---
    print("--- Préparation Données B (LGBM)... ---")
    X_B_full_raw = fe.create_features_base_plus_cyclical(X_full_base)
    X_B_test_raw = fe.create_features_base_plus_cyclical(test_df_cleaned)
    
    X_B_full_le, X_B_test_le = X_B_full_raw.align(X_B_test_raw, join='inner', axis=1)
    
    X_B_full_le = X_B_full_le.copy()
    X_B_test_le = X_B_test_le.copy()
    
    numerical_cols_B = X_B_full_le.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_B = X_B_full_le.select_dtypes(include='object').columns.tolist()
    
    # Imputation Manuelle Simple (pour LE)
    for col in numerical_cols_B:
        median_val = X_B_full_le[col].median()
        X_B_full_le[col] = X_B_full_le[col].fillna(median_val)
        X_B_test_le[col] = X_B_test_le[col].fillna(median_val)
        
    for col in categorical_cols_B:
        mode_series = X_B_full_le[col].mode()
        mode_val = "Unknown" if mode_series.empty else mode_series[0]
        X_B_full_le[col] = X_B_full_le[col].fillna(mode_val)
        X_B_test_le[col] = X_B_test_le[col].fillna(mode_val)
        
        # Label Encoding
        le_col = LabelEncoder()
        combined_series = pd.concat([X_B_full_le[col], X_B_test_le[col]]).astype(str).unique()
        le_col.fit(combined_series)
        
        X_B_full_le[col] = le_col.transform(X_B_full_le[col].astype(str))
        X_B_test_le[col] = le_col.transform(X_B_test_le[col].astype(str))
        
    # Définir les noms des caractéristiques catégorielles pour LGBM
    categorical_features_names_B = categorical_cols_B.copy()
    if 'arrival_date_month' not in categorical_features_names_B:
         categorical_features_names_B.append('arrival_date_month') # 'arrival_date_month' est objet, mais aligné
         
    # 'arrival_date_month' a été encodé par LabelEncoder, donc il est ok
    # Ajustons la liste pour n'inclure que les colonnes qui existent
    categorical_features_names_B = [col for col in categorical_features_names_B if col in X_B_full_le.columns]


    # --- 3.3 Données C (P3-Challenger): Pour CatBoost (FE Interaction + Pas d'encodage) ---
    print("--- Préparation Données C (CatBoost)... ---")
    X_C_full_raw = fe.create_features_catboost(X_full_base)
    X_C_test_raw = fe.create_features_catboost(test_df_cleaned)
    
    X_C_full, X_C_test = X_C_full_raw.align(X_C_test_raw, join='inner', axis=1)

    numerical_cols_C = X_C_full.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_C = X_C_full.select_dtypes(include='object').columns.tolist()

    # Imputation Simple (CatBoost gère bien les NaNs, mais 'Missing' est plus explicite)
    for col in numerical_cols_C:
        median_val = X_C_full[col].median()
        X_C_full[col] = X_C_full[col].fillna(median_val)
        X_C_test[col] = X_C_test[col].fillna(median_val)
        
    for col in categorical_cols_C:
        X_C_full[col] = X_C_full[col].fillna("Missing")
        X_C_test[col] = X_C_test[col].fillna("Missing")

    # Noms des caractéristiques catégorielles pour CatBoost
    categorical_features_names_C = [col for col in categorical_cols_C if col in X_C_full.columns]

    print("--- Pré-traitement terminé. ---")
    
    # Retourner tous les ensembles de données et informations nécessaires
    datasets = {
        'A_rf': (X_A_full, X_A_test),
        'B_lgbm': (X_B_full_le, X_B_test_le, categorical_features_names_B),
        'C_cat': (X_C_full, X_C_test, categorical_features_names_C)
    }
    
    return datasets, y_full

if __name__ == "__main__":
    # Pour tester ce module indépendamment
    print("Test du module data_loader...")
    datasets, y_full = load_and_preprocess_data()
    if datasets:
        print("\n--- Tailles des données ---")
        print(f"y_full: {y_full.shape}")
        
        X_A_full, X_A_test = datasets['A_rf']
        print(f"Données A (RF): Train={X_A_full.shape}, Test={X_A_test.shape}")
        
        X_B_full, X_B_test, cats_B = datasets['B_lgbm']
        print(f"Données B (LGBM): Train={X_B_full.shape}, Test={X_B_test.shape}")
        print(f"   Catégories LGBM: {cats_B}")
        
        X_C_full, X_C_test, cats_C = datasets['C_cat']
        print(f"Données C (CatBoost): Train={X_C_full.shape}, Test={X_C_test.shape}")
        print(f"   Catégories CatBoost: {cats_C}")
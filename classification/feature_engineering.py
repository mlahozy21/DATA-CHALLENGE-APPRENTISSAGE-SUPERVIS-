# --- Fichero: feature_engineering.py ---

import pandas as pd
import numpy as np

# --- 1. Fonction de base (utilisée par les autres) ---
def create_features_lgbm(df):
    """Crée les caractéristiques de base (total_nights, total_guests, adr_per_person, etc.)."""
    df_feat = df.copy()
    
    # Caractéristiques de durée et de groupe
    df_feat['total_nights'] = df_feat['stays_in_weekend_nights'] + df_feat['stays_in_week_nights']
    df_feat['total_guests'] = df_feat['adults'] + df_feat['children'] + df_feat['babies']
    
    # Nettoyage de 'adr' (Average Daily Rate)
    # Trouver la médiane des 'adr' valides ( > 0)
    adr_median = df_feat[df_feat['adr'] > 0]['adr'].median()
    # S'assurer que la médiane n'est pas nulle ou NaN (au cas où)
    if pd.isna(adr_median) or adr_median == 0: 
        adr_median = 0.01 # Mettre une petite valeur pour éviter la division par zéro
    
    df_feat['adr'] = df_feat['adr'].replace(0, adr_median)
    
    # Caractéristiques dérivées
    # Adr par personne (ajouter 1e-6 pour éviter la division par zéro si total_guests est 0)
    df_feat['adr_per_person'] = df_feat['adr'] / (df_feat['total_guests'] + 1e-6)
    df_feat['is_family'] = ((df_feat['children'] > 0) | (df_feat['babies'] > 0)).astype(int)
    
    # Ratio entre le temps d'attente (lead_time) et la durée du séjour
    # (Ajouter 1e-6 pour éviter la division par zéro si total_nights est 0)
    df_feat['lead_vs_stay_ratio'] = df_feat['lead_time'] / (df_feat['total_nights'] + 1e-6)
    
    return df_feat

# --- 2. Fonction pour LGBM (avec caractéristiques cycliques) ---
def create_features_base_plus_cyclical(df):
    """Crée les caractéristiques de base + caractéristiques cycliques (sin/cos) pour le mois et la semaine."""
    df_feat = create_features_lgbm(df)
    
    # Mapper les mois textuels en nombres
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df_feat['arrival_month_num'] = df_feat['arrival_date_month'].map(month_map)
    
    # Caractéristiques cycliques pour le mois
    df_feat['arrival_month_sin'] = np.sin(2 * np.pi * df_feat['arrival_month_num'] / 12)
    df_feat['arrival_month_cos'] = np.cos(2 * np.pi * df_feat['arrival_month_num'] / 12)
    
    # Caractéristiques cycliques pour la semaine (en supposant 53 semaines max)
    df_feat['arrival_date_week_sin'] = np.sin(2 * np.pi * df_feat['arrival_date_week_number'] / 53)
    df_feat['arrival_date_week_cos'] = np.cos(2 * np.pi * df_feat['arrival_date_week_number'] / 53)
    
    return df_feat

# --- 3. Fonction pour CatBoost (avec interactions) ---
def create_features_catboost(df):
    """Crée les caractéristiques de base + interactions de caractéristiques catégorielles."""
    df_feat = create_features_lgbm(df)
    
    # Créer des interactions simples que CatBoost pourrait utiliser
    df_feat['hotel_market_segment'] = df_feat['hotel'].astype(str) + '_' + df_feat['market_segment'].astype(str)
    df_feat['hotel_customer_type'] = df_feat['hotel'].astype(str) + '_' + df_feat['customer_type'].astype(str)
    
    return df_feat
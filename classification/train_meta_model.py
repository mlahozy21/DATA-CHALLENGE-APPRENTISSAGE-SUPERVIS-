# --- Fichero: train_meta_model.py ---

import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression

# Importer nos configurations
import c.config as config

def train_l1_manager():
    """
    Charge les méta-features (L1) et entraîne le gestionnaire final (L1).
    """
    print("--- Démarrage de l'entraînement du Gestionnaire L1 (Méta-Modèle)... ---")

    # --- 1. Charger les Méta-Features (Cache L1) ---
    print("  Étape 1/3 : Chargement des méta-features (X_meta_train) et des cibles (y_full)...")
    try:
        X_meta_train = np.load(config.META_TRAIN_PATH)
        y_full = np.load(config.Y_FULL_PATH)
    except FileNotFoundError as e:
        print(f"Erreur: Fichier de cache L1 non trouvé. {e}")
        print("Veuillez d'abord exécuter 'train_base_models.py' pour générer les méta-features.")
        return

    print(f"  Méta-features chargées : X_meta_train shape = {X_meta_train.shape}")
    print(f"  Cibles chargées : y_full shape = {y_full.shape}")

    # --- 2. Définir le Gestionnaire L1 ---
    print("  Étape 2/3 : Initialisation du gestionnaire L1 (LogisticRegression)...")
    final_estimator = LogisticRegression(**config.final_estimator_params)

    # --- 3. Entraîner et Sauvegarder le Gestionnaire L1 ---
    print("  Étape 3/3 : Entraînement et sauvegarde du méta-modèle...")
    final_estimator.fit(X_meta_train, y_full)
    
    # Sauvegarder le modèle final
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(final_estimator, config.META_MODEL_PATH)
    
    print("\n--- Entraînement L1 Terminé ---")
    print(f"Le méta-modèle final (Gestionnaire L1) a été sauvegardé dans :")
    print(f"{config.META_MODEL_PATH}")

if __name__ == "__main__":
    train_l1_manager()
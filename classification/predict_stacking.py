# --- Fichero: predict_stacking.py ---

import pandas as pd
import numpy as np
import joblib
import os

# Importer nos configurations
import c.config as config

def generate_stacking_submission():
    """
    Charge le méta-modèle L1, les méta-features de test L1,
    prédit les résultats finaux et crée le fichier de soumission.
    """
    print("--- Démarrage du processus de prédiction Stacking (L1)... ---")

    # --- 1. Charger les composants requis ---
    print("  Étape 1/4 : Chargement des composants (Méta-Modèle L1, Méta-Test, IDs, Encodeur)...")
    try:
        # Charger le gestionnaire L1 entraîné
        meta_model = joblib.load(config.META_MODEL_PATH)
        
        # Charger les prédictions L0 sur les données de test (méta-features de test)
        X_meta_test = np.load(config.META_TEST_PATH)
        
        # Charger l'encodeur de label (pour convertir 0,1,2 en 'Canceled', etc.)
        le_y = joblib.load(config.LE_Y_PATH)
        
        # Charger les IDs de test pour le fichier de soumission
        test_ids = joblib.load(config.TEST_IDS_PATH)
        
    except FileNotFoundError as e:
        print(f"Erreur: Fichier modèle ou cache manquant. {e}")
        print("Veuillez exécuter 'train_base_models.py' et 'train_meta_model.py' avant de prédire.")
        return
    
    print(f"  Composants chargés. X_meta_test shape = {X_meta_test.shape}")

    # --- 2. Générer les prédictions finales (encodées) ---
    print("  Étape 2/4 : Génération des prédictions finales...")
    predictions_encoded = meta_model.predict(X_meta_test)

    # --- 3. Décoder les prédictions ---
    print("  Étape 3/4 : Décodage des labels (ex: 0 -> 'Canceled')...")
    predictions_labels = le_y.inverse_transform(predictions_encoded)

    # --- 4. Créer et sauvegarder le fichier de soumission ---
    print(f"  Étape 4/4 : Création du fichier de soumission '{config.SUBMISSION_FILE}'...")
    submission_df = pd.DataFrame({
        'row_id': test_ids, 
        'reservation_status': predictions_labels
    })
    
    submission_df.to_csv(config.SUBMISSION_FILE, index=False)

    print("\n--- Processus de Prédiction Terminé ---")
    print(f"Le fichier de soumission a été sauvegardé avec succès : {config.SUBMISSION_FILE}")

if __name__ == "__main__":
    generate_stacking_submission()
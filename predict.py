# src/predict.py
import pandas as pd
import joblib
import config
import load_data as ld
import warnings
import os

warnings.filterwarnings('ignore')

def run_predictions(model_name):
    """Script principal para generar predicciones."""
    print(f"Iniciando predicciones con el modelo: {model_name.upper()}")
    
    model = None
    preprocessor = None
    train_cols_to_load = []

    # --- 1. Cargar el/los archivo(s) de modelo ---
    
    try:
        if model_name == 'lgbm':
            model_path = config.LGBM_MODEL_PATH
            model = joblib.load(model_path)
        elif model_name == 'catboost':
            model_path = config.CATBOOST_MODEL_PATH
            model = joblib.load(model_path)

    except FileNotFoundError as e:
        print(f"Error: Archivo no encontrado. {e}")
        print(f"Ejecuta 'python -m src.train' para entrenar el modelo '{model_name}' primero.")
        return
    except Exception as e:
        print(f"Error al cargar archivos: {e}")
        return

    # --- 2. Cargar datos de test ---
    print("Cargando datos de test...")
    dftest, row_ids = ld.load_test_data()
    
    # --- 3. Preprocesar y Predecir ---
    print(f"Generando predicciones en {dftest.shape[0]} filas...")
    y_pred = model.predict(dftest)
    
    # --- 4. Crear archivo de submission ---
    submission_df = pd.DataFrame({
        'row_id': row_ids,
        config.VARIABLE_TO_PREDICT: y_pred
    })
    
    # --- 5. Post-procesar (clip) ---
    submission_df[config.VARIABLE_TO_PREDICT] = submission_df[config.VARIABLE_TO_PREDICT].clip(0, 100)
    
    # --- 6. Guardar ---
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"¡Éxito! Archivo de submission guardado en: {config.SUBMISSION_PATH}")

if __name__ == "__main__":
    
    # --- Lógica de pregunta (sin cambios) ---
    model_choice = input("¿Qué modelo quieres usar? (lgbm / catboost): ").strip().lower()
    
    while model_choice not in ['lgbm', 'catboost']:
        print("Error: Opción no válida.")
        model_choice = input("Por favor, escribe 'lgbm' o 'catboost': ").strip().lower()
        
    run_predictions(model_name=model_choice)
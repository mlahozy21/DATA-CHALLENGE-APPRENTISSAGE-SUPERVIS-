# src/train_stack.py
import pandas as pd
import joblib
import warnings
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
import config
import load_data as ld
import numpy as np

def train_meta_model():
    """
    Entrena un meta-modelo (stacking) usando las predicciones
    de los modelos base en el set de validación.
    """
    print("Iniciando entrenamiento del meta-modelo (Stacking)...")
    
    # --- 1. Cargar Datos de Validación (Caché 1 y 2) ---
    try:
        print("Cargando datos (Caché 1 y 2)...")
        X_train, X_test, y_train, y_test= ld.load_train_data(use_cache=True)
        print("Datos de validación cargados.")
    except FileNotFoundError:
        print("Error: No se encontraron los cachés de datos.")
        return

    # --- 2. Cargar Modelos Base ---
    try:
        model_lgbm = joblib.load(config.LGBM_MODEL_PATH)
        model_cat = joblib.load(config.CATBOOST_MODEL_PATH)
        X_train_nocat, X_test_nocat= ld.load_train_data_nocategories(use_cache=True)
        model_kernel = joblib.load(config.KERNEL_MODEL_PATH)
        print("Modelos base cargados.")
    except FileNotFoundError as e:
        print(f"Error: Falta un archivo de modelo base: {e}")
        return

    # --- 3. Generar Predicciones (Out-of-Fold) ---
    print("Generando predicciones OOF (Out-of-Fold)...")
    # Nota: Es mejor usar probabilidades (predict_proba) para modelos lineales 
    # en stacking, pero usaremos las etiquetas como se indica para mantener simple
    # la estructura del DataFrame X_meta_train.
    pred_lgbm_val = model_cat.predict(X_test).ravel()
    pred_cat_val = model_cat.predict(X_test).ravel()
    pred_kernel_val = model_kernel.predict(X_test_nocat).ravel()
    # --- DEBUG: Comprobar longitudes ---

# -----------------------------------
    # --- 4. Crear el Set de Entrenamiento del Meta-Modelo ---
    X_meta_train = pd.DataFrame({
        'lgbm_pred': pred_lgbm_val,
        'cat_pred': pred_cat_val,
        'kernel_pred': pred_kernel_val
    })
    
    # El target es el 'y' real
    y_meta_train = y_test
    
    # --- CAMBIO 2: Inicializar LogisticRegression ---
    

    meta_model = LGBMRegressor(
        n_estimators=75,       # Relativamente pocos árboles
        num_leaves=10,          # Poca complejidad
        learning_rate=0.05,     # Tasa de aprendizaje estándar
        n_jobs=-1,
        random_state=config.RANDOM_STATE
    )
    
    meta_model.fit(X_meta_train, y_meta_train)

    # Mostrar la "importancia" (peso) que le da a cada modelo
    print("Meta-Modelo2 (Regresión Logística) entrenado.")
    print("Pesos de los modelos base (Coeficientes de Regresión Logística):")   
    # Evaluar el meta-modelo en los mismos datos (para referencia)
    y_pred_meta = meta_model.predict(X_meta_train)
    # Usamos average='weighted' para F1-Score multiclase
    meta_r2 = r2_score(y_meta_train, y_pred_meta)
    print(f"\nPuntuación r2 del Meta-Modelo (en validación): {meta_r2:.6f}")
    
    # --- 6. Guardar el Meta-Modelo ---
    joblib.dump(meta_model, config.META_MODEL_PATH)
    print(f"Meta-Modelo2 guardado en: {config.META_MODEL_PATH}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    train_meta_model()

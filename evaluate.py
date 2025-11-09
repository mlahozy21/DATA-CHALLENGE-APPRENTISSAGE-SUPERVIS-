# src/evaluate.py
import joblib
import os
import warnings
from sklearn.metrics import r2_score
import config
import load_data as ld

def run_evaluation():
    """
    Carga todos los modelos entrenados y calcula su R2 Score
    en el conjunto de prueba (validación) correspondiente.
    """
    print("Iniciando pipeline de evaluación...")
    try:
        print("Cargando datos (Caché 1)...")
        X_train, X_test, y_train, y_test, _ = ld.load_train_data(use_cache=True)
    except FileNotFoundError:
        print("Error: No se encontró el caché de datos (Caché 1).")
        print("Por favor, ejecuta 'python -m src.train' al menos una vez para generarlo.")
        return
    
    # Cargar Caché 2 (para SVR)
    try:
        print("Cargando datos (Caché 2)...")
        # X_train_sklearn no lo usaremos, pero X_test_sklearn es vital
        _, X_test_sklearn = ld.load_or_create_sklearn_data(X_train, X_test)
    except FileNotFoundError:
        print("Error: No se encontró el caché de datos (Caché 2).")
        print("Por favor, ejecuta 'python -m src.train' al menos una vez para generarlo.")
        return
        
    print("Datos de validación cargados.")
    
    results = []

    # --- 2. Evaluar LGBM ---
    print("\nEvaluando LGBM...")
    if os.path.exists(config.LGBMinicial_MODEL_PATH):
        model_lgbm = joblib.load(config.LGBMinicial_MODEL_PATH)
        y_pred_lgbm = model_lgbm.predict(X_test)
        score_lgbm = r2_score(y_test, y_pred_lgbm)
        results.append(("LightGBM", score_lgbm))
    else:
        results.append(("LightGBM", "Modelo no encontrado"))

    # --- 3. Evaluar CatBoost ---
    print("Evaluando CatBoost...")
    if os.path.exists(config.CATBOOSTinicial_MODEL_PATH):
        model_cat = joblib.load(config.CATBOOSTinicial_MODEL_PATH)
        y_pred_cat = model_cat.predict(X_test)
        score_cat = r2_score(y_test, y_pred_cat)
        results.append(("CatBoost", score_cat))
    else:
        results.append(("CatBoost", "Modelo no encontrado"))

    # --- 4. Evaluar SVR (Lineal) ---
    print("Evaluando SVR (Lineal)...")
    if os.path.exists(config.SVR_MODEL_PATH):
        model_svr = joblib.load(config.SVR_MODEL_PATH)
        # ¡IMPORTANTE! Usamos X_test_sklearn aquí
        y_pred_svr = model_svr.predict(X_test_sklearn)
        score_svr = r2_score(y_test, y_pred_svr)
        results.append(("SVR (Lineal)", score_svr))
    else:
        results.append(("SVR (Lineal)", "Modelo no encontrado"))
        
    # --- 5. Mostrar Resultados ---
    print("\n" + "="*50)
    print("  Resultados de Evaluación (R2 Score) en Set de Test")
    print("="*50)
    
    for name, score in results:
        if isinstance(score, float):
            print(f"  - {name:<15}: {score:.6f}")
        else:
            print(f"  - {name:<15}: {score}")
    print("="*50)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    run_evaluation()
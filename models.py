import lightgbm as lgbm
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import config as config
import joblib 
import os
import pandas as pd


#LIGHTBGM INICIAL PARA ENCONTRAR HIPERPARAMETROS Y LUEGO LGBM FINAL
def find_optimal_estimators_lgbm(X_train, y_train, X_test, y_test):
    """Usa early stopping para encontrar el n_estimators óptimo para LGBM."""
    print("Buscando n_estimators óptimo para LightGBM...")
    
    eval_model = LGBMRegressor(**config.LGBM_EVAL_PARAMS)
    
    eval_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgbm.early_stopping(100, verbose=False)],
        categorical_feature=config.CAT_VARIABLES
    )
    
    optimal_n = eval_model.best_iteration_
    if optimal_n is None or optimal_n < 50:
        optimal_n = 50 
        
    print(f"Número óptimo de estimadores LGBM encontrado: {optimal_n}")
    
    y_pred = eval_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"LGBM (Evaluación) R2: {r2:.4f} | MSE: {mse:.4f}")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(eval_model, config.LGBMinicial_MODEL_PATH)
    return optimal_n

def train_final_lgbm(X_full, y_full, optimal_n_estimators):
    """Entrena el modelo LGBM final con todos los datos y lo guarda."""
    print("Entrenando modelo LGBM final...")

    final_params = config.LGBM_FINAL_PARAMS.copy()
    final_params['n_estimators'] = optimal_n_estimators
    
    lgbm_model = LGBMRegressor(**final_params)
    
    lgbm_model.fit(
        X_full,
        y_full,
        categorical_feature=config.CAT_VARIABLES
    )
    
    print("Modelo final LGBM entrenado.")
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(lgbm_model, config.LGBM_MODEL_PATH)
    print(f"Modelo guardado en: {config.LGBM_MODEL_PATH}")

    return lgbm_model

#CATBOOST INICIAL PARA ENCONTRAR HIPERPARAMETROS Y LUEGO CATBOOST FINAL

def find_optimal_iterations_catboost(X_train, y_train, X_test, y_test):
    """Usa early stopping para encontrar las iteraciones óptimas para CatBoost."""
    print("Buscando iteraciones óptimas para CatBoost...")
    
    cat_model_eval = CatBoostRegressor(**config.CATBOOST_PARAMS)
    
    cat_model_eval.fit(
        X_train,
        y_train,
        cat_features=config.CAT_VARIABLES,
        eval_set=(X_test, y_test),
        early_stopping_rounds=100
    )
    
    optimal_n = cat_model_eval.get_best_iteration()
    if optimal_n is None or optimal_n < 50:
        optimal_n = 50
        
    print(f"Número óptimo de iteraciones CatBoost encontrado: {optimal_n}")
    
    y_pred = cat_model_eval.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"CatBoost (Evaluación) R2: {r2:.4f} | MSE: {mse:.4f}")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(cat_model_eval, config.CATBOOSTinicial_MODEL_PATH)
    return optimal_n

def train_final_catboost(X_full, y_full, optimal_n_iterations):
    """Entrena el modelo CatBoost final con todos los datos y lo guarda."""
    print("Entrenando modelo CatBoost final...")

    final_params = config.CATBOOST_PARAMS.copy()
    final_params['iterations'] = optimal_n_iterations
    final_params['verbose'] = 0 
    final_params.pop('early_stopping_rounds', None) 

    cat_model_final = CatBoostRegressor(**final_params)
    
    cat_model_final.fit(
        X_full,
        y_full,
        cat_features=config.CAT_VARIABLES
    )
    
    print("Modelo final CatBoost entrenado.")
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(cat_model_final, config.CATBOOST_MODEL_PATH)
    print(f"Modelo guardado en: {config.CATBOOST_MODEL_PATH}")

    return cat_model_final



def train_kernel(X_train, y_train, X_test, y_test, gamma: float = 1.0):

    """
    Aplica una aproximación de kernel RBF (Gaussiano) seguida de un Linear SVC para una clasificación rápida.

    Args:
        X_train, X_test, y_train, y_test: Conjuntos de datos.
        gamma (float): Parámetro del kernel RBF.
    """
    
    print("Iniciando clasificación con Aproximación de Kernel Gaussiano (RBFSampler)...")

    # 1. Crear el pipeline: Escalado -> Aproximación RBF -> Clasificador Lineal
    rbf_feature = RBFSampler(gamma='scale', random_state=config.RANDOM_STATE, n_components=2000)
    
    # LinearSVC es una de las opciones más rápidas para clasificación lineal a gran escala.
    linear_svm = LinearSVR(random_state=config.RANDOM_STATE)

    # Nota: Si el problema es multiclase, asegúrate de que LinearSVC lo maneje correctamente (por defecto es One-vs-Rest).
    
    pipeline = Pipeline([       # Normalización es crucial para métodos basados en distancia
        ('rbf_sampler', rbf_feature),      # Aproximación del kernel
        ('linear_svc', linear_svm)         # Clasificador lineal rápido
    ])

    # 2. Entrenamiento
    pipeline.fit(X_train, y_train)

    # 3. Predicción
    y_pred = pipeline.predict(X_test)

    # 4. Evaluación (usando F1-Score ponderado para multiclase)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nResultados de la Clasificación (RBF Sampler + Linear SVC):")
    print(f"F1-Score (weighted): {r2:.4f}")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, config.KERNEL_MODEL_PATH)
    print(f"Modelo guardado en: {config.KERNEL_MODEL_PATH}")
    
    return pipeline
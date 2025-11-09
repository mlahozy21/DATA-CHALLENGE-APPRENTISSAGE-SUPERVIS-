import load_data as ld
import pandas as pd
import models as m
import os
import warnings
import config
def training(): #BASICAMENTE ENTRENA LOS MODELOS SI ESTOS NO HAN SIDO ENTRENADOS
    #LOAD OF DATA
    X_train, X_test, y_train, y_test = ld.load_train_data(use_cache=True)
    print(f"Datos (Caché 1) cargados: {X_train.shape[0]} filas de entrenamiento.")
    # Combinar datos para entrenamiento final (LGBM/CatBoost)
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0) # y_full es el mismo para todos
    #MODELS
    print("\n--- Procesando LightGBM ---")
    if os.path.exists(config.LGBM_MODEL_PATH):
        print(f"Modelo LGBM ya existe. Saltando.")
    else:
        print("Entrenando LightGBM...")
        optimal_n_lgbm = m.find_optimal_estimators_lgbm(X_train, y_train, X_test, y_test)
        print(f"Entrenando modelo final LGBM...")
        m.train_final_lgbm(X_full, y_full, optimal_n_lgbm)

    print("\n--- Procesando CatBoost ---")
    if os.path.exists(config.CATBOOST_MODEL_PATH):
        print(f"Modelo CatBoost ya existe. Saltando.")
    else:
        print("Entrenando CatBoost...")
        optimal_n_cat = m.find_optimal_iterations_catboost(X_train, y_train, X_test, y_test)
        print(f"Entrenando modelo final CatBoost...")
        m.train_final_catboost(X_full, y_full, optimal_n_cat)
    print("\n--- Procesando kernel ---")
    X_train, X_test= ld.load_train_data_nocategories(use_cache=True)
    if os.path.exists(config.KERNEL_MODEL_PATH):
        print(f"Modelo kernel ya existe. Saltando.")
    else:
        print("Entrenando kernel...")
        m.train_kernel(X_train, y_train,X_test,  y_test)
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    training()

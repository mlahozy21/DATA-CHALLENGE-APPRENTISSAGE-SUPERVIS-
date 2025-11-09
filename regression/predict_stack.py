# src/predict_stack.py
import pandas as pd
import joblib
import warnings
import config
import load_data as ld

def run_stacking_prediction():
    """
    Carga todos los modelos base Y el meta-modelo
    para generar la predicción final (stacking).
    """
    print("Iniciando predicción de Stacking...")

    # --- 1. Cargar TODOS los modelos ---
    try:
        model_lgbm = joblib.load(config.LGBM_MODEL_PATH)
        model_cat = joblib.load(config.CATBOOST_MODEL_PATH)
        model_kernel=joblib.load(config.KERNEL_MODEL_PATH)
        meta_model = joblib.load(config.META_MODEL_PATH)
        print("Todos los modelos (base y meta) cargados.")
    except FileNotFoundError as e:
        print(f"Error: Falta un archivo de modelo: {e}")
        print("Asegúrate de haber ejecutado 'python -m src.train' Y 'python -m src.train_stack'")
        return
    except Exception as e:
        print(f"Error al cargar archivos: {e}")
        return

    # --- 2. Cargar datos de test (crudos) ---
    dftest, row_ids = ld.load_test_data()
    dftest_nocat,_ =ld.load_test_data_nocat()
    print("Datos de test cargados.")
    # --- 3. Generar predicciones de modelos base ---
    print("Generando predicciones de modelos base...")
    pred_lgbm = model_cat.predict(dftest).ravel()
    pred_cat = model_cat.predict(dftest).ravel()
    pred_kernel = model_kernel.predict(dftest_nocat).ravel()


    
    # --- 4. Crear el Set de Test del Meta-Modelo ---
    # ¡Importante! El orden de las columnas debe ser el mismo que en train_stack.py
    X_meta_test = pd.DataFrame({
        'lgbm_pred': pred_lgbm,
        'cat_pred': pred_cat,
        'kernel_pred': pred_kernel
    })

    # --- 5. Generar predicción final ---
    print("Generando predicción final con Meta-Modelo...")
    y_pred_final = meta_model.predict(X_meta_test)

    # --- 6. Crear y guardar submission ---
    submission_df = pd.DataFrame({
        'row_id': row_ids,
        config.VARIABLE_TO_PREDICT: y_pred_final
    })
    
    submission_df[config.VARIABLE_TO_PREDICT] = submission_df[config.VARIABLE_TO_PREDICT].clip(0, 100)
    
    # Guardar con un nombre de archivo único
    stack_submission_path = config.SUBMISSION_PATH
    submission_df.to_csv(stack_submission_path, index=False)
    
    print(f"¡Éxito! Archivo de submission de stacking guardado en: {stack_submission_path}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    run_stacking_prediction()

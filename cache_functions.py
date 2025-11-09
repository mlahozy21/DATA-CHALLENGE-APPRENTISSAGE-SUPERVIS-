import config
import os
import pandas as pd
def lookatcache(type,use_cache=True):
    if type=='train_data':
        X_train, X_test, y_train, y_test,cache =None, None, None, None, None
        os.makedirs(config.PROCESSED_DIR, exist_ok=True)
        cache_files_exist = (
            os.path.exists(config.CACHE_TRAIN_X) and
            os.path.exists(config.CACHE_TRAIN_Y) and
            os.path.exists(config.CACHE_TEST_X) and
            os.path.exists(config.CACHE_TEST_Y) 
        )
        if use_cache and cache_files_exist:
            X_train=pd.read_parquet(config.CACHE_TRAIN_X)
            y_train = pd.read_parquet(config.CACHE_TRAIN_Y).squeeze() 
            X_test = pd.read_parquet(config.CACHE_TEST_X)
            y_test = pd.read_parquet(config.CACHE_TEST_Y).squeeze()
            cache='True'
        
        return X_train, X_test, y_train, y_test, cache
    if type=='train_data_nocategories':
        X_train, X_test, cache =None, None, None
        os.makedirs(config.PROCESSED_DIR, exist_ok=True)
        cache_exists = (
        os.path.exists(config.CACHE_NOCATEGORIES_TRAIN_X) and
        os.path.exists(config.CACHE_NOCATEGORIES_TEST_X)
        )
        if cache_exists and use_cache:
            print("Cargando datos (Caché 2) desde .parquet...")
            X_train = pd.read_parquet(config.CACHE_NOCATEGORIES_TRAIN_X)
            X_test = pd.read_parquet(config.CACHE_NOCATEGORIES_TEST_X)
            cache='True'
        return X_train, X_test,cache

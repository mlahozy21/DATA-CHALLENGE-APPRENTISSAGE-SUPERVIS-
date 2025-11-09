import pandas as pd
import os
import config
from sklearn.model_selection import train_test_split
import funciones as f
import cache_functions as cf
import joblib

def load_train_data(use_cache=True):
    #TRY TO USE CACHE
    X_train, X_test, y_train, y_test, cache= cf.lookatcache(type='train_data')
    if cache=='True':
        return X_train, X_test, y_train, y_test
    #IF WE DON'T FIND CACHE
    df=pd.read_csv(config.TRAIN_DATA)
    df=f.transformationscat(df)
    X=df.drop(columns=[config.VARIABLE_TO_PREDICT])
    y=df[config.VARIABLE_TO_PREDICT]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SPLIT_SIZE, 
        random_state=config.RANDOM_STATE
    )
    if config.N_CLUSTERS>0:
        X_train,X_test=f.clustering(X_train,X_test)
    X_train.to_parquet(config.CACHE_TRAIN_X)
    y_train.to_frame().to_parquet(config.CACHE_TRAIN_Y)
    X_test.to_parquet(config.CACHE_TEST_X)
    y_test.to_frame().to_parquet(config.CACHE_TEST_Y)
    return X_train, X_test, y_train, y_test

def load_test_data():
    #WE APPLY THE SAME TRANSFORMATION TO THE TEST FILE
    dftest = pd.read_csv(config.TEST_DATA)
    row_ids = dftest['row_id'] 
    dftest=f.transformationscat(dftest)
    return dftest, row_ids

def load_train_data_nocategories(use_cache=True):
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    # Comprobar si el caché de sklearn existe
    X_train, X_test, cache= cf.lookatcache(type='train_data_nocategories')
    if cache=='True':
        return X_train, X_test
    df = pd.read_csv(config.TRAIN_DATA)
    df=f.transformacionnocat(df)
    X=df.drop(columns=[config.VARIABLE_TO_PREDICT])
    y=df[config.VARIABLE_TO_PREDICT]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SPLIT_SIZE, 
        random_state=config.RANDOM_STATE
    )
    X_train, X_test=f.preprocessing(X_train, X_test)
    return X_train, X_test

def load_test_data_nocat():
    dftest = pd.read_csv(config.TEST_DATA)
    row_ids = dftest['row_id'] 
    dftest=f.transformacionnocat(dftest)
    dftest,_ =f.preprocessing(dftest, dftest)
    return dftest, row_ids

    
    





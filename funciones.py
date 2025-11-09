import re
import config
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from kmodes.kprototypes import KPrototypes
import joblib
import os

def createnewfeatures(df):
    df['voice_instrument_ratio'] = df['speechiness'] / (df['instrumentalness'] + 0.01) 
    df['party_score'] = df['energy'] * df['danceability'] 
    df['calm_score'] = df['acousticness'] * (1 - df['energy'])
    df['feel_good_score'] = df['valence'] * df['danceability']
    df['studio_intensity'] = (1 - df['liveness']) * df['energy']
    return df

def clean_col_names(df):
    """Limpia los nombres de las columnas."""
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    return df

def set_categories(df, cat_features):
    """Convierte columnas a tipo 'category'."""
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

def transformationscat(df):
    if config.DROP_COLS[0] in df.columns:
        df = df.drop(columns=config.DROP_COLS)
    df=createnewfeatures(df)
    df=set_categories(df,config.CAT_VARIABLES)
    df=clean_col_names(df)
    return df

def preprocessing(X_train,X_test):
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = config.CAT_VARIABLES
    numeric_features = [col for col in numeric_features if col not in categorical_features]
    # Pipeline numérico: usa los parámetros FIJOS del config
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    if os.path.exists(config.PROCESSOR_PATH):
        model_path = config.PROCESSOR_PATH
        preprocessor = joblib.load(model_path)
        X_train = preprocessor.transform(X_train)
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        X_train = preprocessor.fit_transform(X_train)
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        joblib.dump(preprocessor, config.PROCESSOR_PATH)
        
    X_test = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    X_train.to_parquet(config.CACHE_NOCATEGORIES_TRAIN_X)
    X_test.to_parquet(config.CACHE_NOCATEGORIES_TEST_X)
    return X_train, X_test

def transformacionnocat(df):
    if config.DROP_COLS[0] in df.columns:
        df = df.drop(columns=config.DROP_COLS)
    df=createnewfeatures(df)
    df=clean_col_names(df)
    return df

def clustering(X_train,X_test):
    # 1. Separación de variables
    col_numericas = [col for col in X_train.columns if col not in config.CAT_VARIABLES]
    data_numerica = X_train[col_numericas].copy()
    data_categorica = X_train[config.CAT_VARIABLES].copy()
    scaler = StandardScaler()
    data_numerica_scaled = scaler.fit_transform(data_numerica)
    data_categorica_np = data_categorica.values
    # Concatenar los arrays (Numéricos escalados | Categóricos)
    data_final = np.hstack((data_numerica_scaled, data_categorica_np))
    # 4. Obtener los índices de las columnas categóricas en el array final
    # Son las últimas `len(col_categoricas)` columnas del array final.
    indices_categoricos = list(range(
        len(col_numericas), 
        len(col_numericas) + len(config.CAT_VARIABLES)
    ))
    
    # 5. Aplicación del Algoritmo K-Prototypes
    kproto = KPrototypes(
        n_clusters=config.N_CLUSTERS, 
        init='Cao', # 'Cao' es un método de inicialización eficiente
        n_init=5, 
        max_iter=5, 
        random_state=config.RANDOM_STATE,
        verbose=1 # Ajustar a 1 para ver el progreso de las iteraciones
    )
    clusters = kproto.fit_predict(data_final, categorical=indices_categoricos)
    data_resultado = X_train.copy()
    data_resultado['Cluster'] = clusters
    # Preparar X_test
    data_numerica_test = X_test[col_numericas].copy()
    data_categorica_test = X_test[config.CAT_VARIABLES].copy()
    data_numerica_scaled_test = scaler.transform(data_numerica_test)
    data_categorica_np_test = data_categorica_test.values
    data_final_test = np.hstack((data_numerica_scaled_test, data_categorica_np_test))
    clusters_test = kproto.predict(data_final_test, categorical=indices_categoricos)
    X_test_clustered = X_test.copy()
    X_test_clustered['Cluster'] = clusters_test
    
    return data_resultado, X_test_clustered

    





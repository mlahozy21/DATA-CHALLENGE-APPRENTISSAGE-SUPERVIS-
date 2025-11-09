#DIRECTORIES FOR USING THE DATASETS
DATA_DIR="data/" 
TRAIN_DATA=DATA_DIR+"train_data.csv"
TEST_DATA=DATA_DIR+"test_data.csv"
SUBMISSION_TEMPLATE=DATA_DIR+"naive_submission.csv"
SUBMISSION_PATH = "submission.csv" # <-- Archivo de salida

#VARIABLES FOR PREPROCESSING
DROP_COLS=['row_id']
CAT_VARIABLES=['track_genre','time_signature', 'key']
VARIABLE_TO_PREDICT="popularity"
RANDOM_STATE=65
TEST_SPLIT_SIZE=0.3
N_CLUSTERS=0

#CACHE PREPROCESSING
PROCESSED_DIR="processedregression/"
CACHE_TRAIN_X=PROCESSED_DIR+"train_X.parquet"
CACHE_TRAIN_Y=PROCESSED_DIR+"train_Y.parquet"
CACHE_TEST_X=PROCESSED_DIR+"test_X.parquet"
CACHE_TEST_Y=PROCESSED_DIR+"test_Y.parquet"
CACHE_NOCATEGORIES_TRAIN_X=PROCESSED_DIR+"train_nocategories_X.parquet"
CACHE_NOCATEGORIES_TEST_X=PROCESSED_DIR+"test_nocategories_X.parquet"
#CACHE MODELS
MODEL_DIR = "modelsregression/"
LGBM_MODEL_PATH = MODEL_DIR + "lgbm_model.joblib"
CATBOOST_MODEL_PATH = MODEL_DIR + "catboost_model.joblib"
LGBMinicial_MODEL_PATH = MODEL_DIR + "lgbminicial_model.joblib"
CATBOOSTinicial_MODEL_PATH = MODEL_DIR + "catboostinicial_model.joblib"
KERNEL_MODEL_PATH = MODEL_DIR + "kernel_model.joblib"
META_MODEL_PATH=MODEL_DIR + "meta_model.joblib"
PROCESSOR_PATH = MODEL_DIR + "processor.joblib" # <-- ¡NUEVO Y MUY IMPORTANTE!

#HYPERPARAMETERS MODELS
LGBM_EVAL_PARAMS= {
    'objective': 'regression',
    'n_estimators': 30000,
    'learning_rate': 0.01,
    'random_state': RANDOM_STATE,
    'reg_lambda': 1.0,
    'reg_alpha': 0.1
}

LGBM_FINAL_PARAMS = {
    'objective': 'regression',
    'learning_rate': 0.01,
    'random_state': RANDOM_STATE,
    'reg_lambda': 1.0,
    'reg_alpha': 0.1
}

CATBOOST_PARAMS = {
    'iterations': 10000,
    'learning_rate': 0.25,
    'loss_function': 'RMSE',
    'eval_metric': 'R2',
    'random_seed': RANDOM_STATE,
    'verbose': 100,
    'allow_writing_files': False,
    'l2_leaf_reg': 3.0
}


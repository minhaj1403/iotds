"""
Configuration settings for the K-EmoPhone stress prediction project.
"""

import pytz

# Random seed for reproducibility
RANDOM_STATE = 42

# Timezone settings
DEFAULT_TZ = pytz.FixedOffset(540)  # GMT+09:00; Asia/Seoul

# Cross-validation settings
CV_SPLITS = 5
CV_SHUFFLE = True

# Model parameters
XGBOOST_PARAMS = {
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss',
    'eval_size': 0.2,
    'early_stopping_rounds': 10,
    'objective': 'binary:logistic',
    'verbosity': 0,
    'learning_rate': 0.01,
    'tree_method': 'hist',
    'device': 'cuda'
}

TABNET_PARAMS = {
    'device_name': 'cuda',
    'n_d': 64,
    'n_a': 64,
    'n_steps': 5,
    'gamma': 1.5,
    'lambda_sparse': 1e-4,
    'verbose': 0,
    'seed': RANDOM_STATE
}

# Feature selection parameters
LASSO_PARAMS = {
    'penalty': 'l1',
    'solver': 'liblinear',
    'C': 1,
    'random_state': RANDOM_STATE,
    'max_iter': 4000
}

LASSO_THRESHOLD = 0.005

# SHAP parameters
SHAP_MAX_SAMPLES = 1000

# Hyperopt parameters
HYPEROPT_MAX_EVALS = 100

# File paths (to be updated based on your setup)
DATA_PATH = './features_stress_fixed_K-EmoPhone.pkl'

# Results directory
RESULTS_DIR = 'results'

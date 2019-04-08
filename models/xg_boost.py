import numpy as np
from xgboost import XGBClassifier

estimator = XGBClassifier(max_depth=5, n_estimators=200, n_jobs=4)
param_grid = {
    'n_estimators': np.linspace(100, 500, num=50, dtype=np.int16),
    'max_depth': np.arange(3, 33),
    'learning_rate': np.linspace(0.1, 0.001, num=5),
    'max_features': ['auto', 'sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}

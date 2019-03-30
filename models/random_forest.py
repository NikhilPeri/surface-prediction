import numpy as np
from sklearn.ensemble import RandomForestClassifier

estimator = RandomForestClassifier()
param_grid = {
    'n_estimators': np.linspace(500, 1500, num=50, dtype=np.int16),
    'min_samples_split': np.linspace(0.01, 1., num=30),
    'max_features': ['auto', 'sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}

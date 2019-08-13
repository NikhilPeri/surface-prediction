import numpy as np
from sklearn.svm import SVC

estimator = SVC(max_iter=1000)
param_grid = {
    'C': np.linspace(0, 1, num=50, dtype=np.float64),
    'degree': np.arange(2, 6),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    'class_weight': [None, 'balanced']
}

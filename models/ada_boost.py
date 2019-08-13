import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

estimator = AdaBoostClassifier()
param_grid = {
    'base_estimator': [
        DecisionTreeClassifier(max_depth=1),
        DecisionTreeClassifier(max_depth=2),
        DecisionTreeClassifier(max_depth=3),
        DecisionTreeClassifier(max_depth=4)
    ],
    'n_estimators': np.linspace(25, 500, num=50, dtype=np.int16),
    'learning_rate': np.linspace(0.1, 0.001, num=5),
}

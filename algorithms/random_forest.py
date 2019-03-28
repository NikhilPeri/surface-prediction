import numpy as np
import pandas as pd
from pipelines.preprocess import build_training, build_test

from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier

train_data = build_training()

labels = train_data['surface'].values
train_data = train_data.filter(regex='avg_*|sum_*|med_*|var_*|min_*|max_*')
features = np.array(train_data[train_data.columns].values.tolist())

estimator = RandomForestClassifier(
    #n_iter_no_change=10,
    #tol=1e-4,
    #class_weight='balanced',
    #bootstrap='True',
    #max_features='auto',
    warm_start=True,
    verbose=1
)
param_grid = {
    'n_estimators': [500, 700, 900, 1100, 1300, 1500],
    'max_depth': [5, 7, 9, 11, 13, 15, 17, 19, 21],
    #'min_samples_split': [2],
    #'min_impurity_decrease': [0, 1e-5]
    'class_weight': [None, 'balanced'],
    'bootstrap': [True, False],
    'max_features': ['auto', 'sqrt', 'log2'],
    #'learning_rate': [0.05],
    #'subsample': [1.0]
}
grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    scoring='accuracy',
    cv=10,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(features, labels)
    #sample_weight=compute_sample_weight('balanced', labels)
#)
# ExtraTreesClassifier (0.94) {'n_estimators': 700, 'min_impurity_decrease': 1e-05}
# GradientBoostingClassifier (0.88) Best Params: {'max_features': 'log2', 'n_estimators': 800, 'learning_rate': 0.05, 'max_depth': 9, 'subsample': 1.0}
# RandomForestClassifier Best Params: {'n_estimators': 800, 'max_depth': 17, 'class_weight': 'balanced'}
#
print 'Best Score: {}'.format(grid_search.best_score_)
print 'Best Params: {}'.format(grid_search.best_params_)

test_features = build_test()
test_features = test_features.filter(regex='avg_*|sum_*|med_*|var_*|min_*|max_*')
features = np.array(test_features[test_features.columns].values.tolist())

labels = grid_search.best_estimator_.predict(features)

submission = pd.DataFrame({'surface': labels})
submission.index.name = 'series_id'
submission.to_csv('data/y_test.csv')

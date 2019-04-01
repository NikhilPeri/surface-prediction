import numpy as np
import pandas as pd
from pipelines.preprocess import build_training, build_test

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFECV

labels = pd.read_csv('data/y_train.csv')
labels = labels['surface'].values

try:
    features = np.load('features.npy')
except Exception as e:
    train_data = build_training()
    train_data = train_data.filter(regex='^avg_|^sum_|^med_|^var_|^min_|^max_|^max_to_min_|^count_')
    features = np.array(train_data[train_data.columns].values.tolist())
    np.save('features', features)
    train_data = None

estimator = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=15,
    max_features='sqrt',
    verbose=3,
    random_state=420
)

selector = RFECV(estimator, step=0.1, cv=5, verbose=3, min_features_to_select=10000, n_jobs=-1)
selector = selector.fit(features, labels)

    #sample_weight=compute_sample_weight('balanced', labels)
#)
# ExtraTreesClassifier (0.94) {'n_estimators': 700, 'min_impurity_decrease': 1e-05}
# GradientBoostingClassifier (0.88) Best Params: {'max_features': 'log2', 'n_estimators': 800, 'learning_rate': 0.05, 'max_depth': 9, 'subsample': 1.0}
# RandomForestClassifier Best Params: {'n_estimators': 800, 'max_depth': 15, max_features='sqrt' 'class_weight': 'balanced'}
#

for x in selector.grid_scores_:
    print x

try:
    features = np.load('test_features.npy')
except Exception as e:
    test_features = build_test()
    test_features = test_features.filter(regex='^avg_|^sum_|^med_|^var_|^min_|^max_|^max_to_min_|^count_')
    features = np.array(test_features[test_features.columns].values.tolist())
    np.save('test_features', features)
    test_features = None

labels = selector.predict(features)

submission = pd.DataFrame({'surface': labels})
submission.index.name = 'series_id'
submission.to_csv('data/y_test.csv')

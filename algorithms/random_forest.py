import numpy as np
import pandas as pd
from pipelines.preprocess import build_training, build_test

from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from evolutionary_search import EvolutionaryAlgorithmSearchCV

labels = pd.read_csv('data/y_train.csv')
labels = labels['surface'].values

try:
    features = np.load('features.npy')
except Exception as e:
    train_data = build_training()
    train_data = train_data.filter(regex='^avg_|^sum_|^med_|^var_|^min_|^max_|^max_to_min_|^count_')
    features = np.array(train_data[train_data.columns].values.tolist())
    np.save('features', features)

estimator = RandomForestClassifier(
    #n_iter_no_change=10,
    #tol=1e-4,
    #class_weight='balanced',
    #bootstrap='True',
    #max_features='auto',
    n_jobs=-1
)
param_grid = {
    'n_estimators': np.linspace(200, 2000, num=20, dtype=int),
    'max_depth': np.arange(5, 20),
    #'min_samples_split': [2],
    #'min_impurity_decrease': [0, 1e-5]
    'class_weight': [None, 'balanced'],
    'bootstrap': [True, False],
    'max_features': ['auto', 'sqrt', 'log2'],
    #'learning_rate': [0.05],
    #'subsample': [1.0]
}

cv = EvolutionaryAlgorithmSearchCV(estimator=estimator,
                                   params=param_grid,
                                   scoring="accuracy",
                                   cv=StratifiedKFold(n_splits=5),
                                   verbose=True,
                                   population_size=6,
                                   gene_mutation_prob=0.50,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=20)
cv.fit(features, labels)
    #sample_weight=compute_sample_weight('balanced', labels)
#)
# ExtraTreesClassifier (0.94) {'n_estimators': 700, 'min_impurity_decrease': 1e-05}
# GradientBoostingClassifier (0.88) Best Params: {'max_features': 'log2', 'n_estimators': 800, 'learning_rate': 0.05, 'max_depth': 9, 'subsample': 1.0}
# RandomForestClassifier Best Params: {'n_estimators': 800, 'max_depth': 17, 'class_weight': 'balanced'}
#
import pdb; pdb.set_trace()
print 'All Params: {}'.format(cv.cv_results_)
print 'Best Score: {}'.format(cv.best_score_)
print 'Best Params: {}'.format(cv.best_params_)

test_features = build_test()
test_features = test_features.filter(regex='^avg_|^sum_|^med_|^var_|^min_|^max_|^max_to_min_|^count_')
features = np.array(test_features[test_features.columns].values.tolist())

labels = grid_search.best_estimator_.predict(features)

submission = pd.DataFrame({'surface': labels})
submission.index.name = 'series_id'
submission.to_csv('data/y_test.csv')

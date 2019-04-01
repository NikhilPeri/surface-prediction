import numpy as np
import pandas as pd
from pipelines.preprocess import build_training, build_test

from sklearn.linear_model import MultiTaskElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

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

estimators = [
    ('random_forest', RandomForestClassifier(
        n_estimators=800,
        max_depth=15,
        max_features='sqrt',
        class_weight='balanced'
    )),
    ('knn', KNeighborsClassifier(

    )),
    ('gradient_boost', GradientBoostingClassifier(

    )),
    ('extra_trees', ExtraTreesClassifier(

    )),
    ('ada_boost', AdaBoostClassifier(

    )),
    ('SVC', LinearSVC(

    )),
    ('Bayes', GaussianNB(

    )),
    ('', MultiTaskElasticNet(

    ))
]

cv_scores=[]
for name, estimator in estimators:
    print 'Fitting {}'.format(name)
    cv_score = cross_validate(
        estimator, features, labels,
        cv=StratifiedKFold(n_splits=5, shuffle=True),
        n_jobs=-1,
        verbose=1
    )
    cv_scores.append(cv_score['test_score'].mean())
    estimator.fit(features, labels)

model = VotingClassifier(
    estimators=estimators,
    weights=cv_scores**2,
    voting='soft'
)

try:
    features = np.load('test_features.npy')
except Exception as e:
    test_features = build_test()
    test_features = test_features.filter(regex='^avg_|^sum_|^med_|^var_|^min_|^max_|^max_to_min_|^count_')
    features = np.array(test_features[test_features.columns].values.tolist())
    np.save('test_features', features)
    test_features = None


labels = model.predict(features)

submission = pd.DataFrame({'surface': labels})
submission.index.name = 'series_id'
submission.to_csv('data/voting_ensemble/y_test.csv')

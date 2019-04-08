import os
import sys
import importlib
from joblib import dump

from sklearn.model_selection import StratifiedKFold, cross_validate
from pipelines.preprocess import fetch_training

def train(estimator, cv_score='accuracy'):
    features, labels = fetch_training()

    cv = cross_validate(
        estimator, features, labels,
        cv=StratifiedKFold(n_splits=5),
        n_jobs=-1,
        return_estimator=True
    )

    return cv['test_score'].mean(), cv['estimator']

if __name__ == '__main__':
    model_module = sys.argv[1].replace('/', '.').replace('.py', '')
    model_module = importlib.import_module(model_module)

    score, estimator = train(model_module.estimator)

    score = round(score, 4)
    print 'Best Score: {}'.format(score)

    model_dir = sys.argv[1].replace('.py', '')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    dump(estimator, os.path.join(model_dir, 'model-{}.list'.format(score)))

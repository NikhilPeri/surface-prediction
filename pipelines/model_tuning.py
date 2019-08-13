import os
import sys
import importlib
from joblib import dump
import multiprocessing
import numpy as np

from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.model_selection import StratifiedKFold

from pipelines.preprocess import fetch_training

DEFAULT_SEARCH_KWARGS = {
    'scoring':'accuracy',
    'cv': StratifiedKFold(n_splits=5),
    'population_size': 10,
    'gene_mutation_prob': 0.10,
    'gene_crossover_prob': 0.5,
    'tournament_size': 3,
    'generations_number': 5,
    'verbose': True,
    'n_jobs': multiprocessing.cpu_count()
}

def tune(estimator, param_grid, return_search=False, **kwargs):
    merged_kwargs = DEFAULT_SEARCH_KWARGS.copy()
    merged_kwargs.update(kwargs)

    cv = EvolutionaryAlgorithmSearchCV(
        estimator=estimator,
        params=param_grid,
        **merged_kwargs
    )

    features, labels = fetch_training()
    feature = np.load('data/rfe/X_train.npy')
    cv.fit(features, labels)

    if return_search:
        return cv
    return cv.best_score_, cv.best_estimator_

if __name__ == '__main__':
    model_module = sys.argv[1].replace('/', '.').replace('.py', '')
    model_module = importlib.import_module(model_module)

    score, estimator = tune(model_module.estimator, model_module.param_grid)

    print 'Best Score: {}'.format(score)
    print 'Best Params: {}'.format(estimator.get_params())

    model_dir = sys.argv[1]
    if not os.path.exists(model_dir):
        os.makedir(model_dir)

    dump(estimator, os.path.join(model_dir, round(score, 4)))

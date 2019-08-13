import os
import sys
from joblib import load
import pandas as pd

from sklearn.ensemble import VotingClassifier
from pipelines.preprocess import fetch_test, fetch_training

if __name__ == '__main__':
    estimator = load(sys.argv[1])[0]

    #features, labels = fetch_training()
    #estimator.fit(features, labels)

    features = fetch_test()
    import pdb; pdb.set_trace()
    predictions = estimator.predict(features)

    submission = pd.DataFrame({'surface': predictions})
    submission.index.name = 'series_id'
    submission.to_csv(os.path.join(os.path.dirname(sys.argv[1]), 'y_test.csv'))

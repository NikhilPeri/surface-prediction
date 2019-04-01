import sys
import pandas as pd

from scipy.spatial.distance import squareform
from itertools import combinations

def load_submissions(submissions):
    for i, s in enumerate(submissions):
        print 'Loading Predictions {}: {}'.format(i, s)
        s = pd.read_csv(s)
        s = s.rename(columns={'surface': 'surface_{}'.format(i)})
        submissions[i] = s

    submissions = reduce(lambda left, right: pd.merge(left,right,on='series_id'), submissions)
    return submissions

if __name__ == '__main__':
    submissions = load_submissions(sys.argv[1:])
    submissions['surface'] = submissions.filter(regex='^surface').mode(axis='columns')[0]
    submissions = submissions[['series_id', 'surface']]
    submissions.to_csv('data/y_test.csv', index=False)

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

def percent_similarity(pred_0, pred_1):
    return (pred_0 == pred_1).sum() / float(pred_0.count())

def column_similarities(df, similarity_func=percent_similarity):
    similarities = []
    for col_0, col_1 in combinations(df.columns, 2):
        similarities.append(percent_similarity(
            df[col_0], df[col_1]
        ))
    return squareform(similarities)
    
if __name__ == '__main__':
    submissions = load_submissions(sys.argv[1:])
    similarity_matrix = column_similarities(submissions.filter(regex='^surface_'))
    print similarity_matrix

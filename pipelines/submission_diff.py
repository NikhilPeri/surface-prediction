import pandas as pd
import sys

def load_submissions(submissions):
    submissions = [pd.read_csv(s) for s in submissions]
    submissions = reduce(lambda left, right: pd.merge(left,right,on='series_id'), submissions)
    return submissions

if __name__ == '__main__':
    sub = load_submissions(sys.argv[1:])
    import pdb; pdb.set_trace()

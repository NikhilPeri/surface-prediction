import numpy as np
from sklearn.naive_bayes import BernoulliNB

estimator = BernoulliNB()
param_grid = {
    'alpha': np.linspace(0, 1, num=50, dtype=np.float64),
}

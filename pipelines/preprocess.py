import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.preprocessing import OneHotEncoder

FEATURE_COLUMNS = [
    'orientation_W',
    'orientation_X',
    'orientation_Y',
    'orientation_Z',
    'angular_velocity_X',
    'angular_velocity_Y',
    'angular_velocity_Z',
    'linear_acceleration_X',
    'linear_acceleration_Y',
    'linear_acceleration_Z'
]

def quaternion_to_euler(w, x, y, z):
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        X = np.degrees(np.arctan2(t0, t1))

        t2 = 2.0 * (w * y - z * x)
        Y = np.degrees(np.arcsin(t2))

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        Z = np.degrees(np.arctan2(t3, t4))

        return X, Y, Z

def process_group(samples):
    samples.sort_values('measurement_number', inplace=True)
    samples.reset_index(inplace=True, drop=True)

    measurement = {}
    for fc in FEATURE_COLUMNS:
        measurement[fc] = samples[fc].values

    measurement['euler_X'], measurement['euler_Y'], measurement['euler_Z'] = quaternion_to_euler(
        measurement['orientation_W'], measurement['orientation_X'],
        measurement['orientation_Y'], measurement['orientation_Z']
    )

    for col, values in measurement.items():
        measurement['fft_' + col] = np.fft.rfft(values)

    return measurement

def group_measurements(features):
    pool = Pool(processes=4)

    features.sort_values('series_id', inplace=True)
    grouped_measurements = [ features.iloc[v] for k, v in features.groupby('series_id').groups.items() ]
    grouped_measurements = pool.map(process_group, grouped_measurements)
    return pd.DataFrame(grouped_measurements).rename_axis('series_id')

def build_training():
    labels = pd.read_csv('data/y_train.csv')
    features = pd.read_csv('data/X_train.csv')

    features = group_measurements(features)
    features.reset_index(level=0, inplace=True)
    features = pd.merge(features, labels, on='series_id', how='inner')

    return features

def build_test():
    features = pd.read_csv('data/X_test.csv')
    features = group_measurements(features)
    return features

if __name__ == '__main__':
    build_training().head()

import numpy as np
import pandas as pd
import math

from multiprocessing import Pool
from sklearn.preprocessing import OneHotEncoder
from scipy import signal

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

    feature = {}
    for fc in FEATURE_COLUMNS:
        feature[fc] = samples[fc].values
        feature['abs_' + fc] = np.abs(samples[fc].values)

    feature['norm_quat'] = (feature['orientation_X']**2 + feature['orientation_Y']**2 + feature['orientation_Z']**2 + feature['orientation_W']**2)
    feature['mod_quat'] = (feature['norm_quat'])**0.5
    feature['norm_X'] = feature['orientation_X'] / feature['mod_quat']
    feature['norm_Y'] = feature['orientation_Y'] / feature['mod_quat']
    feature['norm_Z'] = feature['orientation_Z'] / feature['mod_quat']
    feature['norm_W'] = feature['orientation_W'] / feature['mod_quat']

    # Convert quaternion to euler
    feature['euler_X'], feature['euler_Y'], feature['euler_Z'] = quaternion_to_euler(
        feature['orientation_W'], feature['orientation_X'],
        feature['orientation_Y'], feature['orientation_Z']
    )

    feature['total_angular_velocity'] = (feature['angular_velocity_X'] ** 2 + feature['angular_velocity_Y'] ** 2 + feature['angular_velocity_Z'] ** 2) ** 0.5
    feature['total_linear_acceleration'] = (feature['linear_acceleration_X'] ** 2 + feature['linear_acceleration_Y'] ** 2 + feature['linear_acceleration_Z'] ** 2) ** 0.5
    feature['acc_vs_vel'] = feature['total_linear_acceleration'] / feature['total_angular_velocity']
    feature['total_angle'] = (feature['euler_X'] ** 2 + feature['euler_Y'] ** 2 + feature['euler_Z'] ** 2) ** 0.5
    feature['angle_vs_acc'] = feature['total_angle'] / feature['total_linear_acceleration']
    feature['angle_vs_vel'] = feature['total_angle'] / feature['total_angular_velocity']

    # First Derivative
    for col, values in feature.items():
        feature['gradient_' + col] = np.gradient(values)
        feature['abs_gradient_' + col] = np.abs(feature['gradient_' + col])
        feature['second_gradient_' + col] = np.gradient(feature['gradient_' + col])
        feature['abs_second_gradient_' + col] = np.gradient(feature['abs_gradient_' + col])

    # Frequency Domain Features
    for col, values in feature.items():
        fft = np.fft.fft(values)
        feature['mag_fft_' + col] = np.absolute(fft)
        feature['phase_fft_' + col] = np.angle(fft)
        feature['power_fft_' + col] = np.power(feature['mag_fft_' + col], 2)
        feature['power_density_fft_' + col] = feature['power_fft_' + col] / np.sum(feature['power_fft_' + col])

    # Peak Detection
    for col, values in feature.items():
        feature['peak_indicies_' + col], _ = signal.find_peaks(values)
        feature['peak_widths_' + col], feature['peak_height_' + col], _, _ = signal.peak_widths(values, feature['peak_indicies_' + col])
        feature['peak_prominences_' + col], _, _ = signal.peak_prominences(values, feature['peak_indicies_' + col])

    stats = {}
    # Signal Statistics
    for col, values in feature.items():
        if values.shape[0] == 0:
            stats['avg_' + col] = 0.
            stats['sum_' + col] = 0.
            stats['var_' + col] = 0.
            stats['med_' + col] = 0.
            stats['min_' + col] = 0.
            stats['max_' + col] = 0.
            stats['max_to_min_' + col] = 0.
            if col.startswith('peak_indicies_'):
                stats['count_' + col] = 0.
        else:
            stats['avg_' + col] = np.average(values)
            stats['sum_' + col] = np.sum(values)
            stats['var_' + col] = np.var(values)
            stats['med_' + col] = np.median(values)
            stats['min_' + col] = np.min(values)
            stats['max_' + col] = np.max(values)
            stats['max_to_min_' + col] = np.max(values) - np.min(values)
            if col.startswith('peak_indicies_'):
                stats['count_' + col] = values.shape[0]

    return stats

def group_measurements(features):
    pool = Pool(processes=4)

    features.sort_values('series_id', inplace=True)
    grouped_measurements = [ features.iloc[v] for k, v in features.groupby('series_id').groups.items() ]
    grouped_measurements = pool.map(process_group, grouped_measurements)
    pool.close()

    return pd.DataFrame(grouped_measurements).rename_axis('series_id')

def build_training():
    labels = pd.read_csv('data/y_train.csv')
    features = pd.read_csv('data/X_train.csv')

    features = group_measurements(features)
    features.reset_index(level=0, inplace=True)
    features = pd.merge(features, labels, on='series_id', how='inner')

    return features

def fetch_training():
    try:
        features = np.load('tmp/features.npy')
    except Exception as e:
        train_data = build_training()
        train_data = train_data.filter(regex='^avg_|^sum_|^med_|^var_|^min_|^max_|^max_to_min_|^count_')
        features = np.array(train_data[train_data.columns].values.tolist())
        np.save('tmp/features', features)

    features = np.concatenate([
        features,
        np.load('data/fft_embedding/X_train.npy'),
        np.load('data/signal_embedding/X_train.npy'),
        np.load('data/wavelet_embedding/X_train.npy')
    ], axis=1)

    labels = pd.read_csv('data/y_train.csv')
    labels = labels['surface'].values

    return features, labels

def fetch_test():
    try:
        features = np.load('tmp/test_features.npy')
    except Exception as e:
        train_data = build_training()
        train_data = train_data.filter(regex='^avg_|^sum_|^med_|^var_|^min_|^max_|^max_to_min_|^count_')
        features = np.array(train_data[train_data.columns].values.tolist())
        np.save('tmp/test_features', features)

    features = np.concatenate([
        features,
        np.load('data/fft_embedding/X_test.npy'),
        np.load('data/signal_embedding/X_test.npy'),
        np.load('data/wavelet_embedding/X_test.npy')
    ], axis=1)

    return features

def build_test():
    features = pd.read_csv('data/X_test.csv')
    features = group_measurements(features)
    return features

if __name__ == '__main__':
    build_training().head()

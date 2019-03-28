import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, BatchNormalization, Dropout, GaussianNoise, Activation, Flatten
from keras.optimizers import SGD, Adadelta
from sklearn.preprocessing import OneHotEncoder
from pipelines.train import train_model
from pipelines.preprocess import build_training

FEATURE_COLUMNS = [
    'real_fft_euler_X',
    'real_fft_euler_Y',
    'real_fft_euler_Z',
    'real_fft_angular_velocity_X',
    'real_fft_angular_velocity_Y',
    'real_fft_angular_velocity_Z',
    'real_fft_linear_acceleration_X',
    'real_fft_linear_acceleration_Y',
    'real_fft_linear_acceleration_Z',
    'imag_fft_euler_X',
    'imag_fft_euler_Y',
    'imag_fft_euler_Z',
    'imag_fft_angular_velocity_X',
    'imag_fft_angular_velocity_Y',
    'imag_fft_angular_velocity_Z',
    'imag_fft_linear_acceleration_X',
    'imag_fft_linear_acceleration_Y',
    'imag_fft_linear_acceleration_Z',
]

def build_features(train_data):
    return np.array(train_data[FEATURE_COLUMNS].values.tolist())

def build_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(9, activation='tanh'))

    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
    model.summary()

    return model

if __name__ == '__main__':
    train_data = build_training()
    features = build_features(train_data)

    label_encoder = OneHotEncoder(sparse=False)
    labels = label_encoder.fit_transform(train_data['surface'].values.reshape(-1, 1))

    train_model(build_model(features.shape[1:]), features, labels)

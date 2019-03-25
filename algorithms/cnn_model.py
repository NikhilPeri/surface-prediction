import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, BatchNormalization, Dropout, GaussianNoise, Activation, Flatten

from sklearn.preprocessing import OneHotEncoder
from pipelines.train import train_model
from pipelines.preprocess import build_training

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

def build_features(train_data):
    return np.array(train_data[FEATURE_COLUMNS].values.tolist())

def build_model(input_shape):
    model = Sequential()

    model.add(GaussianNoise(0.1, input_shape=input_shape))
    model.add(Conv1D(16, 4, padding='valid', data_format='channels_first'))
    model.add(Conv1D(16, 4, padding='valid', data_format='channels_first'))
    model.add(MaxPooling1D(pool_size=2, data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv1D(32, 4, padding='valid', data_format='channels_first'))
    model.add(Conv1D(32, 4, padding='valid', data_format='channels_first'))
    model.add(MaxPooling1D(pool_size=2, data_format='channels_first'))
    #model.add(Conv1D(64, 4, strides=2, padding='valid', data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv1D(64, 4, padding='valid', data_format='channels_first'))
    model.add(Conv1D(64, 4, padding='valid', data_format='channels_first'))
    model.add(MaxPooling1D(pool_size=2, data_format='channels_first'))
    #model.add(Conv1D(128, 4, strides=2, padding='valid', data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.125))

    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(9, activation='tanh'))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()

    return model

if __name__ == '__main__':
    train_data = build_training()
    features = build_features(train_data)

    label_encoder = OneHotEncoder(sparse=False)
    labels = label_encoder.fit_transform(train_data['surface'].values.reshape(-1, 1))

    train_model(build_model(features.shape[1:]), features, labels)

import pandas as pd
import numpy as np
from scipy import signal
from multiprocessing import Pool

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Model

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

def compute_signal(samples):
    samples.sort_values('measurement_number', inplace=True)
    samples.reset_index(inplace=True, drop=True)

    return samples[FEATURE_COLUMNS].values


def build_model(input_shape):
    model = Sequential()

    model.add(Conv1D(8, 2, padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv1D(8, 2, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(16, 2, padding='same', activation='relu'))
    model.add(Conv1D(16, 2, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(32, 2, padding='same', activation='relu'))
    model.add(Conv1D(32, 2, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(9, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
        metrics=['categorical_accuracy']
    )
    model.summary()

    return model

if __name__ == '__main__':
    train_labels = pd.read_csv('data/y_train.csv')
    train_features = pd.read_csv('data/X_train.csv')
    val_labels = pd.read_csv('data/voting/y_test_0.72.csv')
    val_features = pd.read_csv('data/X_test.csv')

    pool = Pool(processes=4)

    val_features.sort_values('series_id', inplace=True)
    val_features = [ val_features.iloc[v] for k, v in val_features.groupby('series_id').groups.items() ]
    val_features = pool.map(compute_signal, val_features)
    val_features = np.array(val_features)

    train_features.sort_values('series_id', inplace=True)
    train_features = [ train_features.iloc[v] for k, v in train_features.groupby('series_id').groups.items() ]
    train_features = pool.map(compute_signal, train_features)
    train_features = np.array(train_features)

    pool.close()

    label_encoder = OneHotEncoder(sparse=False)
    val_labels = label_encoder.fit_transform(val_labels['surface'].values.reshape(-1, 1))
    train_labels = label_encoder.fit_transform(train_labels['surface'].values.reshape(-1, 1))

    class_weight = train_labels.sum()/(9*train_labels.sum(axis=0))

    checkpoint = ModelCheckpoint('data/signal_embedding/model.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')

    model = build_model(train_features.shape[1:])
    model.fit(
        train_features,
        train_labels,
        shuffle=True,
        epochs=100,
        batch_size=30,
        class_weight={ k:v for k,v in enumerate(class_weight) },
        callbacks=[checkpoint],
        #validation_split=0.2,
        #validation_datap=(val_features, val_labels)
        validation_data=(train_features, train_labels)
    )

    model = load_model('data/fft_embedding/model.h5')
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    embeddings = model.predict(train_features)
    np.save('data/signal_embedding/X_train', embeddings)
    embeddings = model.predict(val_features)
    np.save('data/signal_embedding/X_test', embeddings)

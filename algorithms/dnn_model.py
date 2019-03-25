import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, BatchNormalization, Dropout, GaussianNoise, Activation, Flatten

def build_model(input_shape):
    model = Sequential()

    model.add(GaussianNoise(0.1, input_shape=input_shape))

    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.25))
    model.add(Dense(9, activation='tanh'))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.summary()

    return model

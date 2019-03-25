import numpy as np
import os
from datetime import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler, ReduceLROnPlateau

def train_model(model, features, labels):
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=2)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    os.mkdir('tmp')
    save_checkpoint = ModelCheckpoint(
        'tmp/weights-{val_loss:.2f}.hdf5',
        monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=False, mode='auto', period=1
    )
    model.fit(
        features,labels,
        epochs=3,
        batch_size=10,
        validation_split=0.2,
        shuffle=True,
        callbacks=[early_stop, lr_reducer, save_checkpoint],
        verbose=2
    )

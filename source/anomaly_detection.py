from datetime import timedelta
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# seasonality = \
# SinusoidalSeasonality(amplitude=100, period=timedelta(days=3)) \
# + SinusoidalSeasonality(amplitude=10, period=timedelta(days=1))
# noise = RedNoise(mean=0, std=10, correlation=0.5)
# timeseries = seasonality + noise


import pandas as pd
colnames=['timestamp','value'] 
df = pd.read_csv('train.csv',names=colnames, parse_dates=True, index_col="timestamp") 

import numpy as np
import pandas as pd
import keras
from keras import layers
from matplotlib import pyplot as plt

training_mean = df.mean()
training_std = df.std()
df_training_value = (df - training_mean) / training_std
print("Number of training samples:", len(df_training_value))


TIME_STEPS = int(24*7)
#TIME_STEPS = int(72)


# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


x_train = create_sequences(df_training_value)
print("Training input shape: ", x_train.shape)

n_steps = x_train.shape[1]
n_features = x_train.shape[2]
model = keras.Sequential(
    [
        layers.Input(shape=(n_steps, n_features)),
        layers.Conv1D(filters=32, kernel_size=15, padding='same', data_format='channels_last',
            dilation_rate=1, activation="linear"),
        layers.LSTM(
            units=25, activation="tanh", name="lstm_1", return_sequences=False
        ),
        layers.RepeatVector(n_steps),
        layers.LSTM(
            units=25, activation="tanh", name="lstm_2", return_sequences=True
        ),
        layers.Conv1D(filters=32, kernel_size=15, padding='same', data_format='channels_last',
            dilation_rate=1, activation="linear"),
        layers.TimeDistributed(layers.Dense(1, activation='linear'))
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

history = model.fit(
    x_train,
    x_train,
    epochs=100,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, mode="min")
    ],
)








# Photonic LSTM emulator experiments in tensorflow

# Import Helpers
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pandas as pd
from data_handling import univariate_data, import_climate_data, create_time_steps, show_plot

print('Imports Complete \n')

# set figure params for consistency
mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['axes.grid'] = False

# Import data
# This is the 'Weather Time Series Dataset'
# Because it's what the tutorial uses
df = pd.read_csv('jena_climate.csv')
# Split data into training and validation dataset
TRAIN_SPLIT = 300000
# Set the 'random seed' to ensure reproducibility
tf.random.set_seed(13)


# Extract a singular  feature - temperature
uni_data = df['T (degC)']
uni_data.index = df['Date Time']
uni_data.plot().get_figure().savefig('univariate_temperature_data.png')
uni_data = uni_data.values # Clean the data to leave only the temp data

# Gotta normalize the data 4 training~~
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()
uni_data = (uni_data-uni_train_mean)/uni_train_std

# Create multiple datasets to train the model
univariate_past_history = 20
univariate_future_target = 0
# Above variables are used to determine the history and prediction tgt size

# Create univariate data that fits with TF's data requirements
x_train_uni, y_train_uni = univariate_data(uni_data, 0 , TRAIN_SPLIT,
    univariate_past_history, univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
    univariate_past_history, univariate_future_target)

print('Data Cleaning Complete \n')

# Use TF.data to shuffle, batch, and cache the dataset
BATCH_SIZE = 256
BUFFER_SIZE = 10000
train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

print('Data Formatted For TF \n')

# Create a model - vanilla LSTM
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])
simple_lstm_model.compile(optimizer='adam', loss='mae')
print('Vanilla LSTM TF Model Compiled \n')

# Crete a model - No tanh LSTM - this is to compare the impact of activation
pos_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units = 8 , activation = 'sigmoid',
        recurrent_activation='sigmoid', input_shape = x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])
pos_lstm_model.compile(optimizer='adam', loss='mae')
print('No-TANH LSTM TF Model Compiled \n')

# Train the models
EVALUATION_INTERVAL = 150
EPOCHS = 4

simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
    steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_univariate,
    validation_steps=50)
print('Vanilla LSTM Model Trained \n')

pos_lstm_model.fit(train_univariate, epochs=EPOCHS,
    steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_univariate,
    validation_steps=50)
print('no-TANH LSTM Model Trained \n')

print('Generating Predictions Based on Validation Dataset \n')
# Generate Predictions
p_vanilla = simple_lstm_model.predict(x_val_uni)
p_notanh = pos_lstm_model.predict(x_val_uni)
# Plot Prediction Accuracy
fig, ax = plt.subplots(1, 1)
ax.set(title='Prediction Accuracy', xlabel = 'Target Value', ylabel = 'Predicted Value')
van_preds = ax.plot(y_val_uni, p_vanilla, ',b', label='Vanilla LSTM Prediction')
pos_preds = ax.plot(y_val_uni, p_notanh, ',r', label='No-TANH LSTM Prediction')
ax.legend()
plt.savefig('AssessTFPredictions.png')

print('Plot of predicted values generated')

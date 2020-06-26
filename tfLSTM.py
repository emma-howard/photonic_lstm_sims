# Photonic LSTM emulator experiments in tensorflow

# Import Helpers
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pandas as pd

# set figure params for consistency
mpl.rcParams['figure.figsize'] = (8,6)
mpl.rcParams['axes.grid'] = False

# Import data
# This is the 'Weather Time Series Dataset'
# Because it's what the tutorial uses

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)
# Split data into training and validation dataset
TRAIN_SPLIT = 300000
# Set the 'random seed' to ensure reproducibility
tf.random.set_seed(13)

# Part 1 - Forecast a univariate time series

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

# FUNCTIONS YAY

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range (start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data to (histroy_size,1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import tensorflow.keras.metrics as tf_metrics
import logging

def create_model(input_shape, optimizer='Adam', loss='mean_squared_error')-> Sequential:
    """
    Simple function to create sequential model.
    Model is compiled with Adam optimizer and mean squared loss function
    :param input_shape -> tuple int
    :param: optimizer (str)
    :param: loss (str)
    :return: compiled sequential model
    """
    new_model = Sequential()
    new_model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    new_model.add(Dropout(0.2))
    new_model.add(LSTM(units=50, return_sequences=True))
    new_model.add(Dropout(0.2))
    new_model.add(LSTM(units=50))
    new_model.add(Dropout(0.2))
    new_model.add(Dense(units=1))  # prediction of closing value
    new_model.compile(optimizer='Adam', loss='mean_squared_error',
                      metrics=[tf_metrics.MeanSquaredError(),tf_metrics.MeanAbsolutePercentageError()])
    return new_model
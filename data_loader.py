import pandas_datareader as pd_web
import datetime as dt
import numpy as np
import logging

def create_training_tensors(window_size, data):
    """
    Create data tensors for being ready to be consumed by LSTM network for trainig
    :param window_size:
    :param data:
    :return: touple x_train tensor, y_train_tensor
    """
    logging.info(f"Input data shape {data.shape}")
    # x is our learning data
    x_train = []
    # y is what we want to predict
    y_train = []

    for x in range(window_size, len(data)):
        # 60 values from 0
        x_train.append(data[x - window_size:x, 0])
        y_train.append(data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    # add one more dimentsion
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    logging.info(f"Out x-data shape {x_train.shape}")
    logging.info(f"Out y-data shape {y_train.shape}")

    return x_train, y_train


def fetch_data(company_ticker, training_period_y, testing_period_y, from_day):
    """
    Fetch training data from yahoo and return dataframes with daily closing/opening prices.

    :param company_ticker:
    :param training_period_y:
    :param testing_period_y:
    :param from_day:
    :return: 2 dataframes, training data and testing data. Columns: Open, Close, Volume of daily stock prices, index Date
    """
    training_delta = dt.timedelta(days=365*(training_period_y+testing_period_y))
    testing_delta = dt.timedelta(days=365*(testing_period_y))
    start_dt = dt.datetime(2012, 1, 1)
    end_dt = dt.datetime(2020, 1, 1)

    training_data = pd_web.DataReader(company_ticker, 'yahoo', from_day-training_delta, from_day-testing_delta)
    testing_data = pd_web.DataReader(company_ticker, 'yahoo', from_day-testing_delta, from_day)

    logging.info(f"Created testing (from {training_data.index[0]} to {training_data.index[-1]}) "
                 f" and training data (from {testing_data.index[0]} to {testing_data.index[-1]}). ")
    return training_data, testing_data

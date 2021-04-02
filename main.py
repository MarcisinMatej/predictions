import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pd_web
import datetime as dt
import math


from sklearn.preprocessing import MinMaxScaler
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

def plot_actual_vs_predicted(predicted_prices, actual_prices, ticker_symbol, x_labels=None):
    """
    Create plot of predicted vs actual with pyplot
    :param predicted:
    :param actual:
    :param ticker_symbol: for title
    :param x_labels: list of all labels (dates) for our data to be plotted. We will plot every 30 day/labels
    :return:
    """
    plt.plot(actual_prices, color='black', label=f"{ticker_symbol} price")
    plt.plot(predicted_prices, color='blue', label=f"{ticker_symbol} predicted price")
    plt.title(f'LSTM predictions vs actual prices {ticker_symbol} closing stock price')
    plt.xlabel=("Traiding dates")
    if x_labels is not None:
        # ceil, because we need +1 due to range, however we need to avoid situation if we have divisible length of array
        ticks_cnt = math.ceil(len(x_labels)/30)
        new_ticks = [i*30 for i in range(ticks_cnt)]
        # append last day
        new_ticks.append(len(predicted_prices))
        new_labels = [x_labels[i*30] for i in range(ticks_cnt)]
        # append last day value
        new_labels.append(x_labels[-1])
        plt.xticks(ticks=new_ticks,
                   labels=new_labels,
                   rotation=90)
    plt.show()


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

def run_trainig_and_testing(company = "FB", training_window = 60,
                            trainig_period_y=8, testing_period_y=1,
                            prediction_day=dt.datetime.now() ):
    """
    Train LSTM model and predict stock prices
    :param company -> company ticker
    :param training_window -> how many days in past we want to use for prediction, learning window for the model in days

    :return: None
    """
    # Load data for model
    training_data, testing_data = fetch_data(company, trainig_period_y, testing_period_y, prediction_day)
    # prepare data
    # scaler is object which will scale whole series values to defined feature_range interval
    scaler = MinMaxScaler(feature_range=(0,1))
    # we are insterested only in the closing values
    # optionaly we can use AdjClose -> adjusted for splits
    scaled_data = scaler.fit_transform(training_data['Close'].values.reshape(-1,1))

    # x is our learning data
    x_train, y_train = create_training_tensors(training_window, scaled_data)

    # build the model
    model = create_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, epochs=5, batch_size=32)

    # for future we can save the model here and load for future

    """
    TESTING MODEL ACCURACY
    """

    # load test data

    tst_start_dt = dt.datetime(2020,1,1)
    tst_end_dt = dt.datetime(2021,1,1)


    actual_prices = testing_data['Close'].values
    total_dataset = pd.concat((training_data['Close'], testing_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(testing_data) - training_window:].values
    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.fit_transform(model_inputs)

    # make prediction on test data

    x_test, y_test = create_training_tensors(training_window, model_inputs)

    predicted_prices = model.predict(x_test)
    # predicted prices are scaled to (0-1) so we need to rescale them to actual prices
    predicted_prices = scaler.inverse_transform(predicted_prices)

    plot_actual_vs_predicted(predicted_prices,actual_prices, company,
                             x_labels=[date.strftime("%d-%m-%y") for date in  testing_data.index] )

    # predict next day
    real_data = [model_inputs[len(model_inputs) + 1 - training_window: len(model_inputs)+1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    print(scaler.inverse_transform(real_data[-1]))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Next day value is: {prediction}")

    print("===================================")
    print(model.summary())
    print("===================================")
    print("Evaluating model")
    result = model.evaluate(x_test, y_test)
    print(dict(zip(model.metrics_names, result)))


    # TODO
    # experiment with next days predictions


if __name__ == '__main__':
    run_trainig_and_testing()

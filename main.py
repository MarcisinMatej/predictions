import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import logging

from data_loader import *
from predicition_model import *
from visualization import *

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

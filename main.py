import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pd_web
import datetime as dt


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load data
company = "FB"
start_dt = dt.datetime(2012,1,1)
end_dt = dt.datetime(2020,1,1)

data = pd_web.DataReader(company, 'yahoo', start_dt, end_dt)

# prepare data
# scaler is object which will scale whole series values to defined feature_range interval
scaler = MinMaxScaler(feature_range=(0,1))
# we are insterested only in the closing values
# optionaly we can use AdjClose -> adjusted for splits
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# how many days in past we want to use for prediction, learning window for the model in days
training_window = 60
# x is our learning data
x_train = []
# y is what we want to predict
y_train = []

for x in range(training_window, len(scaled_data)):
    # 60 values from 0
    x_train.append(scaled_data[x - training_window:x, 0])
    y_train.append(scaled_data[x, 0])

# conver to numpy for model

x_train, y_train = np.array(x_train), np.array(y_train)
# add one more dimentsion
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# build the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) # prediction of closing value

model.compile(optimizer='Adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# for future we can save the model here and load for future

"""
TESTING MODEL ACCURACY
"""

# load test data

tst_start_dt = dt.datetime(2020,1,1)
tst_end_dt = dt.datetime(2021,1,1)

tst_data = pd_web.DataReader(company, 'yahoo', tst_start_dt, tst_end_dt)
actual_prices = tst_data['Close'].values
total_dataset = pd.concat((data['Close'], tst_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(tst_data) - training_window:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.fit_transform(model_inputs)

# make prediction on test data

x_test = []
for x in range(training_window, len(model_inputs)):
    x_test.append(model_inputs[x - training_window:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test , (x_test.shape[0],x_test.shape[1],1))

predicted_prices = model.predict(x_test)
# predicted prices are scaled to (0-1) so we need to rescale them to actual prices
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.plot(actual_prices, color='black', label=f"{company} price")
plt.plot(predicted_prices, color='blue', label=f"{company} predicted price")
plt.title(f'LSTM predictions vs actual prices {company} closing stock price')
plt.show()

# predict next day
real_data = [model_inputs[len(model_inputs) + 1 - training_window: len(model_inputs)+1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
print(scaler.inverse_transform(real_data[-1]))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Next day value is: {prediction}")

# TODO
# experiment with next days predictions
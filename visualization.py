
import math
import matplotlib.pyplot as plt

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

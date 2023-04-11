from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from os.path import exists
from Algorithm.lib import get_yahoo_data
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop

import os
import pandas as pd
import numpy as np
import datetime

from Algorithm.lib.InitCentersRandom import InitCentersRandom
from Algorithm.lib.RBFLayer import RBFLayer

global model, y_plot_pred, sc_x, x_test


def train_model(currency_o, currency_d, fecha_ini, fecha_fin):
    makemydir('./Data/')
    file_exists = exists('./Data/yahoo_data.dat')

    if file_exists is False:
        yahooData = get_yahoo_data.YahooData(fecha_ini, fecha_fin)
        historical_data = yahooData.getYahooData(currency_o + currency_d + '=X')
        historical_data.to_pickle('./Data/yahoo_data.dat')
    else:
        print('Reading existing file...')
        historical_data = pd.read_pickle('./Data/yahoo_data.dat')

    yahoo_variables = historical_data.to_numpy()[:, [0, 1, 2]]
    yahoo_close_price = historical_data.to_numpy()[:, 3]
    global model, sc_x, x_test

    x_train, x_test, y_train, y_test = train_test_split(yahoo_variables, yahoo_close_price, train_size=0.80,
                                                        shuffle=False)

    sc_x = MinMaxScaler(feature_range=(0, 1))
    sc_y = MinMaxScaler(feature_range=(0, 1))

    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.fit_transform(x_test)

    y_train = sc_y.fit_transform(y_train.reshape(-1, 1))
    y_test = sc_y.fit_transform(y_test.reshape(-1, 1))

    model = Sequential()
    rbflayer = RBFLayer(10, y_train, initializer=InitCentersRandom(x_train), betas=0.008, input_shape=(3,))
    model.add(rbflayer)
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer=RMSprop())

    model.fit(x_train, y_train,
              batch_size=30,
              epochs=250,
              verbose=0)

    y_pred = model.predict(x_test)
    y_plot = sc_y.inverse_transform(y_test.reshape(-1, 1))
    global y_plot_pred
    y_plot_pred = sc_y.inverse_transform(y_pred.reshape(-1, 1))

    # print('x_test', x_test)
    # print('pred', y_plot_pred)

    dates = historical_data.index
    dates_plot = np.array(dates[-y_test.size:].values).astype('datetime64[D]')

    # print(y_plot)
    # print(y_plot_pred)

    return y_plot_pred, y_plot, dates_plot


# dir is not keyword
def makemydir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def make_prediction(prediction_days=3):
    global sc_x, x_test, y_plot_pred, model

    x_plot_p = sc_x.inverse_transform(x_test)
    sc_x_p = MinMaxScaler(feature_range=(1, 2))
    sc_y_p = MinMaxScaler(feature_range=(1, 2))

    x_plot_p = sc_x_p.fit_transform(x_plot_p)
    sc_y_p.fit_transform(y_plot_pred)

    last_prediction = x_plot_p[-1][-1]

    for i in range(0, prediction_days):
        last_result = [last_prediction, last_prediction * 0.02 + last_prediction,
                       last_prediction - last_prediction * 0.02]
        last_result = np.array(last_result).reshape(1, -1)
        x_plot_p = np.append(x_plot_p[1:], last_result, axis=0)

        last_prediction = x_plot_p[-1][-1]

    prediction = sc_y_p.inverse_transform(model.predict(x_plot_p))

    return prediction[-prediction_days:].ravel(), add_day_to_dates(prediction_days)


def add_day_to_dates(prediction_days):
    _dates = np.array([])
    lastDate = datetime.datetime.now()
    for i in range(prediction_days):
        newDate = pd.to_datetime(lastDate) + pd.DateOffset(days=i + 1)
        _dates = np.append(_dates, newDate)
    return _dates.astype('datetime64[D]')

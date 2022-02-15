import json
from datetime import datetime
from os import mkdir, path
from typing import Dict

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from ta import momentum, trend

from utils import load_configs

# Binance API reference
# https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data


class Preprocess:

    _BASE_BINANCE_URL = 'https://api.binance.com/api/v3/klines?symbol=<symbol>&interval=<interval>&startTime=<start_time>'

    def __init__(self) -> None:
        configs = load_configs()
        self.raw_data_dir = configs['data']['raw_data_dir']
        self.processed_data_dir = configs['data']['processed_data_dir']
        self.end_time = configs['binance']['end_time']
        self.symbols = configs['binance']['symbols']
        self.intervals = configs['binance']['intervals']

    def _get_json_response(self, url: str) -> Dict:
        return json.loads(requests.get(url).text)

    def _drop_col(self, df: pd.DataFrame, name: str) -> None:
        df.drop([name], axis=1, inplace=True)

    def _check_if_nan_and_fill(self, df: pd.DataFrame) -> None:
        if df.isnull().values.any():
            null_cols = df.columns[df.isnull().any()]
            df[null_cols].isnull().sum()

            # print('Dataset contains null values')
            # print(data[data.isnull().any(axis=1)][null_cols].head())

            df.fillna(method='ffill', inplace=True)

    def load_processed_data(self, fname: str) -> pd.DataFrame:
        folder_path = self.raw_data_dir + '/' + self.processed_data_dir
        file_path = folder_path + '/' + fname + '.json'
        return pd.read_json(file_path, orient='records')

    def fetch_kline_data(self, symbol: str, interval: str) -> pd.DataFrame | str:
        # Binance limit is 500 records, max 1200 requests/minute

        fname = 'binance_' + symbol + '_' + interval + '.json'
        fpath = path.join(self.raw_data_dir, fname)
        if not path.isdir(self.raw_data_dir):
            mkdir(self.raw_data_dir)

        if not path.isfile(fpath):
            print(f'Downloading data for {symbol}, interval {interval}...')

            # skip first 24h of price data
            url = self._BASE_BINANCE_URL.replace('<symbol>', symbol).replace('<interval>', interval)
            first_timestamp = self._get_json_response(url.replace('<start_time>', '0'))[0][0]  # first timestamp in json
            day_in_millis = 86400000
            response = self._get_json_response(url.replace('<start_time>', str(first_timestamp + day_in_millis)))

            # new start time is the previous end timestamp, 500 is the limit/max
            start_time = response[-1][0]
            end_time = self.convert_date(self.end_time, to_timestamp=True)
            while start_time < end_time:
                new_response = self._get_json_response(url.replace('<start_time>', str(start_time)))
                # omit the first element as it is equal to the last on the previous list
                response = response + new_response[1:]
                start_time = response[-1][0]

            with open(fpath, 'w') as f:
                json.dump(response, f, sort_keys=True, indent=4, ensure_ascii=False)

            return pd.DataFrame(response), fname

        else:
            print('Retrieving from file...')
            return pd.read_json(fpath), fname

    def process_data(self, df: pd.DataFrame, fname: str) -> pd.DataFrame:
        fname = fname.split('binance_')[1]
        fpath = path.join(self.raw_data_dir, self.processed_data_dir)
        file_path = fpath + '/' + fname
        if not path.isdir(fpath):
            mkdir(fpath)

        if not path.isfile(file_path):
            # remove any rows with null values
            df = df.dropna()

            # from binance-api-docs: https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#klinecandlestick-data
            col_names = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume',
                         'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
            df.columns = col_names

            # drop unnecessary columns
            col_drop_names = ['Volume', 'Close Time', 'Number of trades',
                              'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
            df.drop(col_drop_names, axis=1, inplace=True)

            # Quote Asset Volume is volume in base currency
            df.rename(columns={'Quote Asset Volume': 'Volume'}, inplace=True)

            # remove rows after end time for last 500 records batch
            end_time = self.convert_date(self.end_time, to_timestamp=True)
            df = df[df['Open Time'] <= end_time]

            # sort by ascending date
            df = df.sort_values(by='Open Time')

            print('Calculating TA indicators...')
            df = self.calculate_ta(df)

            # save to disk
            with open(file_path, 'w') as f:
                out = df.to_json(orient='records')
                f.write(out)

        else:
            print('Retrieving from file...')
            df = pd.read_json(file_path, orient='records')

        return df

    def calculate_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        def moving_average(data_col: pd.DataFrame, n: int) -> pd.Series:
            ma = data_col.rolling(window=n).mean()
            ma.fillna(0, inplace=True)
            return ma

        # Trend Indicators
        # Moving Average (MA)
        # df['MA_50'] = moving_average(df['Close'], 50)
        # df['MA_200'] = moving_average(df['Close'], 200)

        # Exponential Moving Average (EMA)
        df['EMA13'] = trend.ema_indicator(df['Close'], window=13, fillna=True)
        df['EMA200'] = trend.ema_indicator(df['Close'], window=200, fillna=True)

        # Momentum Indicators
        # Relative Strength Index (RSI)
        df['RSI'] = momentum.rsi(df['Close'], window=14, fillna=True)

        # Volatility Indicators
        # Bollinger Bands (BB)
        # df['BB_H'] = volatility.bollinger_hband_indicator(df['Close'], n=20, ndev=2, fillna=True)
        # df['BB_L'] = volatility.bollinger_lband_indicator(df['Close'], n=20, ndev=2, fillna=True)

        # remove first 200 elements (EMA/MA 200 is nan)
        df = df[200:]

        return df

    def convert_date(self, val: int, to_timestamp: bool) -> pd.DataFrame:
        if to_timestamp:
            dt = datetime.strptime(val, '%d.%m.%Y %H:%M:%S')
            millis_time = dt.timestamp() * 1000
            return int(millis_time)
        else:
            # time units are milliseconds
            date_col = pd.to_datetime(val, unit='ms')
            return date_col

    def scale_minmax(self, df: pd.DataFrame) -> MinMaxScaler:
        # BB_H, BB_L, RSI are good
        cols = [name for name in df.columns if name in ['BB_H', 'BB_L', 'RSI']]
        tmp_df = pd.DataFrame()
        for col in cols:
            tmp_df[col] = df[col]

        # rescale to [0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
        scaler.fit_transform(df.values)

        # replace BB_H, BB_L and RSI with original values
        for col in cols:
            df[col] = tmp_df[col]

        return scaler

    def scale_zero_base(self, df: pd.DataFrame) -> pd.DataFrame | pd.Series:
        # closing price base for inverse scaling
        close_base = df.loc[:, 'Close'].iloc[0]

        # BB_H, BB_L, RSI are good
        norm_cols = [name for name in df.columns if name not in [
            'BB_H', 'BB_L', 'RSI']]
        for col in norm_cols:
            # normalise against the 1st element for each window
            tmp_base = df.loc[:, col].iloc[0]
            df.loc[:, col] = (df.loc[:, col] / tmp_base) - 1

        return df, close_base

    def create_inputs_minmax(self, data: pd.DataFrame, x_win_size: int = 50, y_win_size: int = 1) -> np.ndarray | np.ndarray | MinMaxScaler:
        # can store 2x in memory compared to float64
        tmp_data = data.astype('float32')

        self._drop_col(tmp_data, name='Open Time')
        self._check_if_nan_and_fill(tmp_data)

        # BB_H, BB_L are in the range [0,1]
        # RSI is oscillator [0, 100] --> [0,1]
        tmp_data[['RSI']] = tmp_data[['RSI']] / 100

        scaler = self.scale_minmax(tmp_data)

        x_inputs = []
        y_inputs = []
        i = 0
        while (i + x_win_size + y_win_size) <= len(tmp_data):
            # e.g. x[0:50] y[50:51]
            x_win_data = tmp_data[i: i + x_win_size]
            y_win_data = tmp_data['Close'][i + x_win_size: i + x_win_size + y_win_size]

            # to numpy array
            x_win_arr = np.array(x_win_data)
            y_win_arr = np.array(y_win_data)
            x_inputs.append(x_win_arr)
            y_inputs.append(y_win_arr)

            i = i + 1

        x_inputs = np.array(x_inputs)
        y_inputs = np.array(y_inputs)
        # reshape for plotting (_,)
        y_inputs = np.reshape(y_inputs, (-1,))
        print('Shape X:', np.shape(x_inputs), 'Shape Y:', np.shape(y_inputs))

        return x_inputs, y_inputs, scaler

    def create_inputs_zero_base(self, data: pd.DataFrame, x_win_size: int = 50, y_win_size: int = 1) -> np.ndarray | np.ndarray | np.ndarray:
        # can store 2x in memory compared to float64
        tmp_data = data.astype('float32')

        self._drop_col(tmp_data, name='Open Time')
        self._check_if_nan_and_fill(tmp_data)

        # BB_H, BB_L are in the range [0,1]
        # RSI is oscillator [0, 100]
        # -- Scale to [0, 2], then shift to [-1, 1] range
        tmp_data[['RSI']] = ((tmp_data[['RSI']] / 100) * 2) - 1

        x_inputs = []
        y_inputs = []
        close_bases = []
        i = 0
        while (i + x_win_size + y_win_size) <= len(tmp_data):
            # create a copy to preserve original data
            window_data = tmp_data[i: (i + x_win_size + y_win_size)].copy()
            window_data, close_base = self.scale_zero_base(window_data)

            # x[0:50] y[50:51]
            x_win_data = window_data[: x_win_size]
            y_win_data = window_data['Close'].iloc[-1]

            # change to numpy array
            x_win_arr = np.array(x_win_data)
            x_inputs.append(x_win_arr)
            y_inputs.append(y_win_data)
            close_bases.append(close_base)

            i = i + 1

        x_inputs = np.array(x_inputs)
        y_inputs = np.array(y_inputs)
        close_bases = np.array(close_bases)

        print('Shape X:', np.shape(x_inputs), 'Shape Y:', np.shape(y_inputs))
        return x_inputs, y_inputs, close_bases

    def create_inputs_reinforcement(self, data: pd.DataFrame, x_win_size: int = 10) -> np.ndarray | np.ndarray | MinMaxScaler:
        # can store 2x in memory compared to float64
        tmp_data = data.astype('float32')

        self._drop_col(tmp_data, name='Open Time')
        self._check_if_nan_and_fill(tmp_data)

        # BB_H, BB_L are in the range [0,1]
        # RSI is oscillator [0, 100] --> [0,1]
        tmp_data[['RSI']] = tmp_data[['RSI']] / 100

        scaler = self.scale_minmax(tmp_data)

        # replace all 0, otherwise can cause division with 0
        tmp_data['Close'].replace(0, method='bfill', inplace=True)

        inputs = []
        closing_prices = []
        i = 0
        while (i + x_win_size) <= len(tmp_data):
            # e.g. x[0:50]
            window = tmp_data[i: (i + x_win_size)]
            # price of the window (1st element) at which trade is executed
            close_price = window.loc[:, 'Close'].iloc[0]

            # to numpy array
            win_arr = np.array(window)
            inputs.append(win_arr)
            closing_prices.append(close_price)

            i = i + 1

        inputs = np.array(inputs)
        closing_prices = np.array(closing_prices)

        print('Shape Inputs:', np.shape(inputs))
        return inputs, closing_prices, scaler

    def split_train_test(self, df: pd.DataFrame, train_set_size: int = 0.8) -> pd.DataFrame | pd.DataFrame:
        train_set = df[:int(train_set_size * len(df))]
        test_set = df[int(train_set_size * len(df)):]
        return train_set, test_set

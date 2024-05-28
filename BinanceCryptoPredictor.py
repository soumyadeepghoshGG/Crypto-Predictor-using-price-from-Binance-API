import os
import itertools
import numpy as np
import pandas as pd
import datetime
import pickle
import warnings
import logging
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from prophet import Prophet
from prophet.plot import plot_components_plotly
from binance.client import Client
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ignore all warnings
warnings.filterwarnings(action='ignore')

# Global variables
ohlcv_options = ["open", "high", "low", "close", "volume"]
interval_options = ['1MINUTE', '3MINUTE', '5MINUTE', '15MINUTE', '1HOUR', '2HOUR', '4HOUR', '1DAY', '1WEEK']
crypto_options = {
    "Bitcoin and Altcoins": [
        {"name": "Bitcoin", "symbol": "BTCUSDT"},
        {"name": "Ethereum", "symbol": "ETHUSDT"},
        {"name": "Ripple", "symbol": "XRPUSDT"},
        {"name": "Litecoin", "symbol": "LTCUSDT"},
        {"name": "Cardano", "symbol": "ADAUSDT"},
        {"name": "Polkadot", "symbol": "DOTUSDT"},
        {"name": "Bitcoin Cash", "symbol": "BCHUSDT"},
        {"name": "Chainlink", "symbol": "LINKUSDT"},
        {"name": "Stellar", "symbol": "XLMUSDT"},
        {"name": "Uniswap", "symbol": "UNIUSDT"}
    ],
    "Tokens": [
        {"name": "Basic Attention Token", "symbol": "BATUSDT"},
        {"name": "Compound", "symbol": "COMPUSDT"},
        {"name": "Aave", "symbol": "AAVEUSDT"},
        {"name": "Synthetix", "symbol": "SNXUSDT"},
        {"name": "Yearn.finance", "symbol": "YFIUSDT"},
        {"name": "Maker", "symbol": "MKRUSDT"},
        {"name": "Dai", "symbol": "DAIUSDT"},
        {"name": "Tether", "symbol": "USDTUSDT"}
    ],
    "Privacy Coins": [
        {"name": "Monero", "symbol": "XMRUSDT"},
        {"name": "Zcash", "symbol": "ZECUSDT"},
        {"name": "Dash", "symbol": "DASHUSDT"},
        {"name": "Verge", "symbol": "XVGUSDT"},
        {"name": "Horizen", "symbol": "ZENUSDT"}
    ],
    "Stablecoins": [
        {"name": "Tether", "symbol": "USDTUSDT"},
        {"name": "USD Coin", "symbol": "USDCUSDT"},
        {"name": "Dai", "symbol": "DAIUSDT"},
        {"name": "TrueUSD", "symbol": "TUSDUSDT"},
        {"name": "Paxos Standard", "symbol": "PAXUSDT"}
    ],
    "Utility Tokens": [
        {"name": "Binance Coin", "symbol": "BNBUSDT"},
        {"name": "Basic Attention Token", "symbol": "BATUSDT"},
        {"name": "Chainlink", "symbol": "LINKUSDT"},
        {"name": "Uniswap", "symbol": "UNIUSDT"},
        {"name": "VeChain", "symbol": "VETUSDT"},
        {"name": "Theta Token", "symbol": "THETAUSDT"},
        {"name": "Crypto.com Coin", "symbol": "CROUSDT"},
        {"name": "Maker", "symbol": "MKRUSDT"},
        {"name": "Compound", "symbol": "COMPUSDT"},
        {"name": "Filecoin", "symbol": "FILUSDT"}
    ],
    "Security Tokens": [
        {"name": "Polymath", "symbol": "POLYUSDT"},
        {"name": "tZERO", "symbol": "TZROPUSDT"},
        {"name": "Harbor", "symbol": "RFRUSDT"},
        {"name": "Securitize", "symbol": "DSXUSDT"},
        {"name": "TokenSoft", "symbol": "TSFTUSDT"}
    ],
    "Meme Coins": [
        {"name": "Dogecoin", "symbol": "DOGEUSDT"},
        {"name": "Shiba Inu", "symbol": "SHIBUSDT"},
        {"name": "DogeCash", "symbol": "DOGECUSDT"},
        {"name": "HuskyToken", "symbol": "HUSKYUSDT"},
        {"name": "SafeMoon", "symbol": "SAFEMOONUSDT"},
        {"name": "Floki Inu", "symbol": "FLOKIUSDT"}
    ],
    "AI Coins": [
        {"name": "SingularityNET", "symbol": "AGIUSDT"},
        {"name": "DeepBrain Chain", "symbol": "DBCUSDT"},
        {"name": "Fetch.ai", "symbol": "FETUSDT"},
        {"name": "Ocean Protocol", "symbol": "OCEANUSDT"},
        {"name": "Numerai", "symbol": "NMRUSDT"},
        {"name": "Matrix AI Network", "symbol": "MANUSDT"},
        {"name": "DAGlabs", "symbol": "DAGUSDT"},
        {"name": "Effect.ai", "symbol": "EFXUSDT"},
        {"name": "Neural Protocol", "symbol": "NRPUSDT"},
        {"name": "AGI Coin", "symbol": "AGIUSDT"}
    ]
}




# Fetch live data from Binance API
class DataFetcher:
    def __init__(self, api_key, api_secret, crypto, interval, n_days_past):
        self.client = Client(api_key, api_secret)
        self.crypto = crypto
        self.interval = interval
        self.n_days_past = n_days_past


    def _get_data_from_API(self):
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = end_time - datetime.timedelta(days=self.n_days_past)

        kline_intervals = {
            '1MINUTE': Client.KLINE_INTERVAL_1MINUTE,
            '3MINUTE': Client.KLINE_INTERVAL_3MINUTE,
            '5MINUTE': Client.KLINE_INTERVAL_5MINUTE,
            '15MINUTE': Client.KLINE_INTERVAL_15MINUTE,
            '1HOUR': Client.KLINE_INTERVAL_1HOUR,
            '2HOUR': Client.KLINE_INTERVAL_2HOUR,
            '4HOUR': Client.KLINE_INTERVAL_4HOUR,
            '1DAY': Client.KLINE_INTERVAL_1DAY,
            '1WEEK': Client.KLINE_INTERVAL_1WEEK
        }

        try:
            kline_data = self.client.get_historical_klines(self.crypto,
                                                        kline_intervals[self.interval],
                                                        start_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
                                                        end_time.strftime('%Y-%m-%d %H:%M:%S %Z'))
        except Exception as e:
            logging.error(f"Error fetching data from API: {e}")
            return None

        df = pd.DataFrame(kline_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                            'close_time', 'quote_asset_volume', 'number_of_trades',
                                            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        
        # Convert numeric columns to float and round to 2 decimal places
        cols =['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
            'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        df[cols] = df[cols].astype(float).round(2)
        
        # Convert timestamp columns to datetime and localize to UTC
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', errors='coerce')
               
        # Convert timestamp columns to Asia/Kolkata timezone
        df['timestamp'] =  df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
        df['close_time'] =  df['close_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')

        return df


    def save_data(self, data, path):
        data.to_csv(path, index=False)
        logging.info(f"Data saved to {path}")


    @staticmethod
    def _create_data_path(crypto, interval, n_days_past):
        return f"data/{crypto} data with interval {interval} for {n_days_past} days.csv"




# Implementation of Facebook Prophet model
class ProphetModel:
    def __init__(self, crypto, interval, n_days_past, n_days_future):
        self.crypto = crypto
        self.interval = interval
        self.n_days_past = n_days_past
        self.n_days_future = n_days_future


    @staticmethod
    def _create_model_path(crypto, model_type, interval, n_days_past):
        return f"model/{model_type} model for {crypto}-{interval}-{n_days_past}d.pkl"


    def fit(self, data, model_path, flag):
        if os.path.exists(model_path) and not flag:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
        else:
            model = Prophet()
            model.fit(data)
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
        return model


    def forecast(self, model):
        interval = self.interval
        n_days_future = self.n_days_future
        frequency, periods = self._determine_frequency_and_periods(interval, n_days_future)
        future_data = model.make_future_dataframe(periods=periods, freq=frequency, include_history=True)
        forecast = model.predict(future_data)
        return forecast


    @staticmethod
    def _determine_frequency_and_periods(interval, n_days_future):
        if interval in ['1MINUTE', '3MINUTE', '5MINUTE', '15MINUTE']:
            return "min", n_days_future * 24 * 60
        elif interval in ['1HOUR', '2HOUR', '4HOUR']:
            return "H", n_days_future * 24
        elif interval == '1DAY':
            return "D", n_days_future
        else:
            return "W", n_days_future // 7


    # Prophet Trend
    def plot_trend(self, data, forecast):
        pio.templates.default = "simple_white"

        actual_trend = go.Scatter(
            x=data['ds'],
            y=data['y'],
            mode='lines',
            line={'color': '#000000', 'width': 1},
            name='Actual'
        )

        yhat = go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            line={'color': '#3bbed7', 'width': 3},
            name='Forecast'
        )

        yhat_upper = go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            hoverinfo='none',
            mode='lines',
            line={'color': 'rgba(255, 255, 255, 0)'},
            fill='tonexty',
            fillcolor='rgba(135, 206, 250, 0.35)',
            name='Confidence Interval'
        )

        yhat_lower = go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            hoverinfo='none',
            mode='lines',
            line={'color': 'rgba(255, 255, 255, 0)'},
            showlegend=False
        )

        layout = go.Layout(
            title={'text': f'<b>{self.crypto} trend using {self.interval} interval data for {self.n_days_past} days - Prophet</b>',
                'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
                'font': {'size': 24, 'color': '#800000'}},
            xaxis={'title': '', 'showgrid': False},
            yaxis={'title': '', 'showgrid': True},
            hovermode='x',
            margin={'t': 50, 'b': 0, 'l': 0, 'r': 0},
            legend={'x': 0.05, 'y': 0.95, 'bgcolor': 'rgba(147, 149, 151, 0.4)', 'borderwidth': 1, 'bordercolor': '#000000',
                    'font': {'size': 16, 'color': 'black'}},
            width=1800,
            height=800,
            annotations=[
                dict(
                    text=f"Date: {datetime.datetime.now().strftime('%d-%m-%Y')}",
                    x=0.99,
                    y=1,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(147, 149, 151, 0.4)",
                    bordercolor="black",
                    borderwidth=1
                )
            ]
        )

        fig = go.Figure(data=[yhat_lower, yhat_upper, yhat, actual_trend], layout=layout)

        # fig.write_image(f"plots/f'{self.crypto} trend using {self.interval} Interval - Prophet ({datetime.datetime.now().strftime('%d-%m-%Y')}).png")
        plotly.offline.plot(fig, filename=f"plots/{self.crypto} trend using {self.interval} interval data for {self.n_days_past} days - Prophet ({datetime.datetime.now().strftime('%d-%m-%Y')}.html")


    # Prophet components
    def plot_components(self, model, forecast):
        fig = plot_components_plotly(model, forecast)

        for axis in fig.layout:
            if 'xaxis' in axis:
                fig.layout[axis].update(showgrid=True)
            if 'yaxis' in axis:
                fig.layout[axis].update(showgrid=True)

        fig.update_layout(
            title={'text': f"<b>{self.crypto} components using {self.interval} interval data for {self.n_days_past} days - Prophet</b>",
                'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
                'font': {'size': 24, 'color': '#800000'}},
            showlegend=False,
            width=1800,
            height=1200,
            annotations=[
                dict(
                    text=f"Date: {datetime.datetime.now().strftime('%d-%m-%Y')}",
                    x=0.01,
                    y=1,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(147, 149, 151, 0.4)",
                    bordercolor="black",
                    borderwidth=1
                )
            ]
        )
        
        plotly.offline.plot(fig, filename=f"plots/{self.crypto} components using {self.interval} interval data for {self.n_days_past} days - Prophet ({datetime.datetime.now().strftime('%d-%m-%Y')}).html")




# Implementation of SARIMAX model
class SARIMAXModel:
    def __init__(self, crypto, interval, n_days_past, n_days_future):
        self.crypto = crypto
        self.interval = interval
        self.n_days_past = n_days_past
        self.n_days_future = n_days_future


    @staticmethod
    def _create_model_path(crypto, model_type, interval, n_days_past):
        return f"model/{model_type} model for {crypto}-{interval}-{n_days_past}d.pkl"


    @staticmethod
    def _determine_frequency_and_periods(interval, n_days_future):
        if interval in ['1MINUTE', '3MINUTE', '5MINUTE', '15MINUTE']:
            return "T", n_days_future * 24 * 60
        elif interval in ['1HOUR', '2HOUR', '4HOUR']:
            return "H", n_days_future * 24
        elif interval == '1DAY':
            return "D", n_days_future
        else:
            return "W", n_days_future // 7


    def fit(self, endog, model_path, flag):
        if os.path.exists(model_path) and not flag:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
        else:
            # Search range for hyperparameters - Non-seasonal parameters
            p = d = q = range(0, 2)
            pdq = list(itertools.product(p, d, q))

            # Search range for hyperparameters - Seasonal parameters: Monthly
            seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

            best_aic = float("inf")
            best_params = None
            best_model = None

            for param in pdq:
                for seasonal_param in seasonal_pdq:
                    try:
                        model = SARIMAX(endog=endog, order=param, seasonal_order=seasonal_param, enforce_stationarity=False, enforce_invertibility=False)
                        results = model.fit()
                        
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_params = (param, seasonal_param)
                            best_model = results
                    except Exception as e:
                        continue

            model = best_model

            # Save the best model
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)

        return model


    def forecast(self, model, data):
        interval = self.interval
        n_days_future = self.n_days_future

        frequency, periods = self._determine_frequency_and_periods(interval, n_days_future)
        
        # Generate in-sample predictions
        forecast_in_sample = model.get_prediction()
        if forecast_in_sample is None:
            return None
        forecast_in_sample = forecast_in_sample.summary_frame()
        forecast_in_sample.drop(columns=['mean_se'], inplace=True)
        forecast_in_sample.rename(columns={"mean": "yhat", "mean_ci_upper": "yhat_upper", "mean_ci_lower": "yhat_lower"}, inplace=True)
        forecast_in_sample.index = data.index

        # Extract forecast data - Out of sample
        forecast_out_of_sample = model.get_forecast(steps=periods)
        if forecast_out_of_sample is None:
            return None
        forecast_index = pd.date_range(start=data.index[-1], periods=periods+1, freq=frequency)[1:]
        forecast_out_of_sample = forecast_out_of_sample.summary_frame()
        forecast_out_of_sample.drop(columns=['mean_se'], inplace=True)
        forecast_out_of_sample.rename(columns={"mean": "yhat", "mean_ci_upper": "yhat_upper", "mean_ci_lower": "yhat_lower"}, inplace=True)
        forecast_out_of_sample.index = forecast_index

        # Arrange columns
        cols = ['yhat', 'yhat_upper', 'yhat_lower']
        forecast_in_sample = forecast_in_sample.loc[:, cols]
        forecast_out_of_sample = forecast_out_of_sample.loc[:, cols]

        # Concatenate forecasts
        forecast = pd.concat((forecast_in_sample, forecast_out_of_sample))

        return forecast


    # Plot trend and forecast
    def plot_trend(self, data, forecast):
        pio.templates.default = "ggplot2"

        actual_trend = go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            line={'color': '#B55A30', 'width': 1},
            name='Actual'
        )

        yhat = go.Scatter(
            x=forecast.index,
            y=forecast['yhat'],
            mode='lines',
            line={'color': '#56C6A9', 'width': 3},
            name='Forecast'
        )

        yhat_upper = go.Scatter(
            x=forecast.index,
            y=forecast['yhat_upper'],
            hoverinfo='none',
            mode='lines',
            line={'color': 'rgba(255, 255, 255, 0)'},
            fill='tonexty',
            fillcolor='rgba(56, 126, 109, 0.35)',
            name='Confidence Interval'
        )

        yhat_lower = go.Scatter(
            x=forecast.index,
            y=forecast['yhat_lower'],
            hoverinfo='none',
            mode='lines',
            line={'color': 'rgba(255, 255, 255, 0)'},
            showlegend=False
        )

        layout = go.Layout(
            title={'text': f'<b>{self.crypto} trend using {self.interval} interval data for {self.n_days_past} days - SARIMAX</b>',
                'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
                'font': {'size': 24, 'color': '#800000'}},
            xaxis={'title': '', 'showgrid': False},
            yaxis={'title': '', 'showgrid': True},
            hovermode='x',
            margin={'t': 50, 'b': 0, 'l': 0, 'r': 0},
            legend={'x': 0.05, 'y': 0.95, 'bgcolor': 'rgba(224, 181, 137, 0.3)', 'borderwidth': 1, 'bordercolor': '#000000',
                    'font': {'size': 16, 'color': 'black'}},
            width=1800,
            height=800,
            annotations=[
                dict(
                    text=f"Date: {datetime.datetime.now().strftime('%d-%m-%Y')}",
                    x=0.99,
                    y=1,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(224, 181, 137, 0.3)",
                    bordercolor="black",
                    borderwidth=1
                )
            ]
        )

        fig = go.Figure(data=[yhat_lower, yhat_upper, yhat, actual_trend], layout=layout)

        plotly.offline.plot(fig, filename=f"plots/{self.crypto} trend using {self.interval} interval data for {self.n_days_past} days - SARIMAX ({datetime.datetime.now().strftime('%d-%m-%Y')}).html")




# Predictor class for automated generation forecasts of different models
class CryptoPredictor:
    crypto_symbols = [coin["symbol"] for coin_type in crypto_options.values() for coin in coin_type]

    def __init__(self, api_key, api_secret, crypto='BTCUSDT', interval='1HOUR', ohlcv='open', n_days_past=365, n_days_future=100, override=False):
        self.data_fetcher = DataFetcher(api_key, api_secret, crypto, interval, n_days_past)
        self.prophet_model = ProphetModel(crypto, interval, n_days_past, n_days_future)
        self.sarimax_model = SARIMAXModel(crypto, interval, n_days_past, n_days_future)
        self.__override = override
        if crypto not in CryptoPredictor.crypto_symbols:
            raise ValueError(f"Crypto not available, please choose from {(', ').join(CryptoPredictor.crypto_symbols)}.")
        else:
            self.crypto = crypto
        if ohlcv not in ohlcv_options:
            raise ValueError(f"OHLCV option not available, please choose from {(', ').join(ohlcv_options)}.")
        else:
            self.ohlcv = ohlcv
        if interval not in interval_options:
            raise ValueError(f"Interval option not available, please choose from: {(', ').join(interval_options)}.")
        else:
            self.interval = interval
        if n_days_past < 1:
            raise ValueError("Number of days must be at least 1.")
        else:
            self.n_days_past = n_days_past
        if n_days_future < 1:
            raise ValueError("Number of days must be at least 1.")
        else:
            self.n_days_future = n_days_future


    def _get_data_path(self):
        return self.data_fetcher._create_data_path(self.crypto, self.interval, self.n_days_past)


    # Get either old data or fetch new data from API
    def fetch_data(self):
        data_path = self._get_data_path()
        __data_changed = False
        
        # Prepare data
        if os.path.exists(data_path):
            if self.__override:
                data = self.data_fetcher._get_data_from_API()
                if data is None:
                    logging.error("Data fetching failed.")
                else:
                    __data_changed = True
                    self.data_fetcher.save_data(data, data_path)
            else:
                data = pd.read_csv(data_path)
        else:
            data = self.data_fetcher._get_data_from_API()
            if data is None:
                logging.error("Data fetching failed.")
            else:
                __data_changed = True
                self.data_fetcher.save_data(data, data_path)
        
        return data, __data_changed
    

    def _get_prophet_model_path(self):
        return self.prophet_model._create_model_path(self.crypto, "prophet", self.interval, self.n_days_past)


    def _get_sarimax_model_path(self):
        return self.sarimax_model._create_model_path(self.crypto, "sarimax", self.interval, self.n_days_past)


    # Forecasting crypto data with prophet model
    def forecast_with_prophet(self):
        data, flag = self.fetch_data()
        model_path = self._get_prophet_model_path()
        
        # Refine data
        data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
        data.rename(columns={'timestamp': 'ds', self.ohlcv: 'y'}, inplace=True)
        data = data[['ds', 'y']]        

        # Use Prophet Model
        prophet_model_instance = self.prophet_model.fit(data, model_path, flag)
        prophet_forecast = self.prophet_model.forecast(prophet_model_instance)
        self.prophet_model.plot_trend(data, prophet_forecast)
        self.prophet_model.plot_components(prophet_model_instance, prophet_forecast)


    # Forecasting crypto data with SARIMAX model
    def forecast_with_sarimax(self):
        ohlcv = self.ohlcv
        data, flag = self.fetch_data()
        model_path = self._get_sarimax_model_path()

        # Refine data
        data['timestamp'] = pd.to_datetime(data['timestamp']).dt.tz_localize(None)
        data.set_index('timestamp', inplace=True)
        data.rename(columns={ohlcv: 'y'}, inplace=True)
        endog = data['y']

        # Use SARIMAX Model
        sarimax_model_instance = self.sarimax_model.fit(endog, model_path, flag)
        sarimax_forecast = self.sarimax_model.forecast(sarimax_model_instance, endog)
        self.sarimax_model.plot_trend(endog, sarimax_forecast)
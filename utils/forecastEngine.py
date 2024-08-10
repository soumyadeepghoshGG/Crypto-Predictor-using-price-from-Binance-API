import os
import itertools
import pickle
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import CryptoBase
import warnings
warnings.filterwarnings(action='ignore')

class ForecastEngine(CryptoBase):
    def __init__(self, 
                 data: pd.DataFrame, 
                 crypto: str, 
                 interval: str, 
                 ohlcv: str, 
                 n_days_past: int, 
                 n_days_future: int, 
                 override: bool = False):
        super().__init__(crypto=crypto,
                         interval=interval,
                         ohlcv=ohlcv,
                         n_days_past=n_days_past, 
                         n_days_future=n_days_future,
                         model='Prophet',  # Placeholder
                         override=override)
        self.data = self._prepare_data(data)

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy(deep=True)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
        df.rename(columns={'timestamp': 'ds', self.ohlcv: 'y'}, inplace=True)
        return df[['ds', 'y']]

    def _get_model_path(self, model_type: str) -> str:
        return self._create_path('model', f"{model_type}_model_for_{self.crypto}-{self.interval}-{self.n_days_past}d.pkl")

    def _determine_frequency_and_periods(self) -> tuple:
        if self.interval in ['1MINUTE', '3MINUTE', '5MINUTE', '15MINUTE']:
            return "min", self.n_days_future * 24 * 60
        elif self.interval in ['1HOUR', '2HOUR', '4HOUR']:
            return "H", self.n_days_future * 24
        elif self.interval == '1DAY':
            return "D", self.n_days_future
        else:
            return "W", self.n_days_future // 7

    def _load_or_train_model(self, model_type: str, train_func):
        model_path = self._get_model_path(model_type)
        if os.path.exists(model_path) and not self.override:
            with open(model_path, 'rb') as file:
                return pickle.load(file)
        else:
            model = train_func()
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
            return model

    def train_prophet(self) -> pd.DataFrame:
        def train_prophet():
            model = Prophet()
            model.fit(self.data)
            return model

        model = self._load_or_train_model('Prophet', train_prophet)
        frequency, periods = self._determine_frequency_and_periods()
        future_data = model.make_future_dataframe(periods=periods, freq=frequency, include_history=True)
        forecast = model.predict(future_data)
        
        return forecast

    def train_sarimax(self) -> pd.DataFrame:
        def _model_tuning():
            p = d = q = range(0, 2)
            pdq = list(itertools.product(p, d, q))
            seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

            best_aic = float("inf")
            best_model = None

            for param in pdq:
                for seasonal_param in seasonal_pdq:
                    try:
                        model = SARIMAX(endog=self.data['y'], order=param, 
                                        seasonal_order=seasonal_param, 
                                        enforce_stationarity=False, enforce_invertibility=False)
                        results = model.fit()

                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_model = results
                    except Exception:
                        continue

            return best_model

        model = self._load_or_train_model('SARIMAX', _model_tuning)
        frequency, periods = self._determine_frequency_and_periods()

        forecast_in_sample = model.get_prediction().summary_frame()
        forecast_out_of_sample = model.get_forecast(steps=periods).summary_frame()

        for df in [forecast_in_sample, forecast_out_of_sample]:
            df.rename(columns={'mean': 'yhat', 'mean_ci_lower': 'yhat_lower', 'mean_ci_upper': 'yhat_upper'}, inplace=True)
            df.drop(columns=['mean_se'], inplace=True)

        forecast = pd.concat([forecast_in_sample[['yhat', 'yhat_lower', 'yhat_upper']], 
                              forecast_out_of_sample[['yhat', 'yhat_lower', 'yhat_upper']]])

        forecast.iloc[0] = [0, 0, 0]
        forecast['ds'] = pd.date_range(start=self.data['ds'].iloc[0], periods=len(forecast), freq=frequency)

        return forecast

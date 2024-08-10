from utils.base import CryptoBase
from utils.dataFetcher import DataFetcher
from utils.forecastEngine import ForecastEngine
from utils.plotter import Plotter
from typing import Optional
import pandas as pd

class CryptoPredictor(CryptoBase):
    def __init__(self, 
                 crypto: Optional[str] = 'BTCUSDT',
                 interval: Optional[str] = '1HOUR',
                 ohlcv: Optional[str] = 'open',
                 n_days_past: Optional[int] = 365,
                 n_days_future: Optional[int] = 30, 
                 model: Optional[str] = 'Prophet',
                 override: Optional[bool] = False):
        super().__init__(crypto=crypto,
                         interval=interval,
                         ohlcv=ohlcv, 
                         n_days_past=n_days_past, 
                         n_days_future=n_days_future, 
                         model=model, 
                         override=override)

    def _initialize_forecast_engine(self) -> ForecastEngine:
        historical_data = self.fetch_data()
        return ForecastEngine(historical_data, 
                              self.crypto, self.interval, self.ohlcv, 
                              self.n_days_past, self.n_days_future, self.override)

    def fetch_data(self) -> pd.DataFrame:
        return DataFetcher(self.crypto, self.interval, self.n_days_past, self.override).fetch_data()

    def show_forecast_df(self) -> pd.DataFrame:
        fe = self._initialize_forecast_engine()
        
        if self.model == 'Prophet':
            return fe.train_prophet()
        elif self.model == 'Sarimax':
            return fe.train_sarimax()
        
    def show_trend(self):
        fe = self._initialize_forecast_engine()
        
        if self.model == 'Prophet':
            forecast_data = fe.train_prophet()
        elif self.model == 'Sarimax':
            forecast_data = fe.train_sarimax()
        
        plt = Plotter(self.fetch_data(), forecast_data, self.crypto, self.interval, self.n_days_past, self.model)
        plt.plot_trend()

    def show_components(self):
        fe = self._initialize_forecast_engine()
        forecast_data = fe.train_prophet()
        
        plt = Plotter(self.fetch_data(), forecast_data, self.crypto, self.interval, self.n_days_past, self.model)
        plt.plot_components()

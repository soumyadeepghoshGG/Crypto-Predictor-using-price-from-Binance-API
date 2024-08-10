import os
import datetime
import pandas as pd
from binance.client import Client
import logging
from typing import Optional
from .base import CryptoBase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pd.options.display.float_format = '{:.8f}'.format

class DataFetcher(CryptoBase):
    def __init__(self, 
                 crypto: str, 
                 interval: str, 
                 n_days_past: int, 
                 override: bool):
        super().__init__(crypto=crypto,
                         interval=interval,
                         ohlcv='open',  # Placeholder
                         n_days_past=n_days_past, 
                         n_days_future=1,  # Placeholder
                         model='Prophet',  # Placeholder
                         override=override)
        self.client = self._initialize_client()

    @staticmethod
    def _initialize_client() -> Client:
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        if not api_key or not api_secret:
            raise ValueError("API key and secret must be set as environment variables.")
        return Client(api_key, api_secret)

    def _get_data_path(self) -> str:
        return self._create_path('data', f"{self.crypto}_data_{self.interval}_{self.n_days_past}days.csv")

    def _get_data_from_API(self) -> Optional[pd.DataFrame]:
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = end_time - datetime.timedelta(days=self.n_days_past)

        kline_intervals = {interval: getattr(Client, f'KLINE_INTERVAL_{interval}') for interval in CryptoBase.interval_options}

        try:
            kline_data = self.client.get_historical_klines(
                self.crypto,
                kline_intervals[self.interval],
                start_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
                end_time.strftime('%Y-%m-%d %H:%M:%S %Z')
            )
        except Exception as e:
            logging.error(f"Error fetching data from API: {e}")
            return None

        df = pd.DataFrame(kline_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        numeric_cols = [
            'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
            'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        ]
        df[numeric_cols] = df[numeric_cols].astype(float).round(8)

        for col in ['timestamp', 'close_time']:
            df[col] = pd.to_datetime(df[col], unit='ms', utc=True).dt.tz_convert('Asia/Kolkata')

        if df.isnull().values.any():
            logging.warning("Data contains missing values.")
        
        return df

    def fetch_data(self) -> Optional[pd.DataFrame]:
        path = self._get_data_path()

        if not self.override and os.path.exists(path):
            logging.info(f"Data already exists at {path}. Skipping download.")
            return pd.read_csv(path)

        df = self._get_data_from_API()

        if df is not None:
            df.to_csv(path, index=False)
            logging.info(f"Data saved to {path}")
        else:
            logging.error("No data fetched to save.")
        
        return df

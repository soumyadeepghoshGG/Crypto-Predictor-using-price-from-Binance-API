import os
import json
from typing import Optional

class CryptoBase:
    crypto_options = []
    interval_options = []
    ohlcv_options = []
    model_options = []

    @classmethod
    def _load_options(cls, json_file: str) -> None:
        with open(json_file, 'r') as f:
            data = json.load(f)
            cls.interval_options = data['interval_options']
            cls.ohlcv_options = data['ohlcv_options']
            cls.crypto_options = [item['symbol'] for sublist in data['crypto_options'].values() for item in sublist]
            cls.model_options = data['model_options']

    def __init__(self, 
                 crypto: str,
                 interval: str,
                 ohlcv: str,
                 n_days_past: int,
                 n_days_future: int,
                 model: str,
                 override: bool):
        if not CryptoBase.interval_options or not CryptoBase.ohlcv_options or not CryptoBase.crypto_options or not CryptoBase.model_options:
            CryptoBase._load_options('crypto_options.json')

        self.crypto = self._validate_crypto(crypto)
        self.interval = self._validate_interval(interval)
        self.ohlcv = self._validate_ohlcv(ohlcv)
        self.n_days_past = self._validate_n_days_past(n_days_past)
        self.n_days_future = self._validate_n_days_future(n_days_future)
        self.override = self._validate_override(override)
        self.model = self._validate_model(model)

    @staticmethod
    def _validate_crypto(crypto: str) -> str:
        if crypto not in CryptoBase.crypto_options:
            raise ValueError(f"Invalid crypto symbol: '{crypto}'. Available options: {CryptoBase.crypto_options}")
        return crypto

    @staticmethod
    def _validate_interval(interval: str) -> str:
        if interval not in CryptoBase.interval_options:
            raise ValueError(f"Invalid interval: '{interval}'. Available options: {CryptoBase.interval_options}")
        return interval

    @staticmethod
    def _validate_ohlcv(ohlcv: str) -> str:
        if ohlcv not in CryptoBase.ohlcv_options:
            raise ValueError(f"Invalid OHLCV option: '{ohlcv}'. Available options: {CryptoBase.ohlcv_options}")
        return ohlcv

    @staticmethod
    def _validate_model(model: str) -> str:
        if model not in CryptoBase.model_options:
            raise ValueError(f"Invalid model: '{model}'. Available options: {CryptoBase.model_options}")
        return model

    @staticmethod
    def _validate_n_days_past(n_days_past: int) -> int:
        if not isinstance(n_days_past, int) or n_days_past <= 0:
            raise ValueError(f"Invalid number of days in the past: {n_days_past}. It must be a positive integer.")
        return n_days_past

    @staticmethod
    def _validate_n_days_future(n_days_future: int) -> int:
        if not isinstance(n_days_future, int) or n_days_future < 0:
            raise ValueError(f"Invalid number of days in the future: {n_days_future}. It must be a non-negative integer.")
        return n_days_future

    @staticmethod
    def _validate_override(override: bool) -> bool:
        if not isinstance(override, bool):
            raise ValueError(f"Invalid override: {override}. It must be a boolean.")
        return override

    @staticmethod
    def _create_directory(directory: str) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)

    @classmethod
    def _create_path(cls, directory: str, filename: str) -> str:
        cls._create_directory(directory)
        return os.path.join(directory, filename)

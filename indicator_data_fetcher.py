import pandas as pd
import numpy as np
import pandas_ta as ta
import os
from pathlib import Path

# Dizin Yolları
BASE_DIR = Path(__file__).resolve().parent 
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RAW_CSV_PATH = os.path.join(DATA_DIR, "BTC_USDT_15m_LABELED.csv")
PROCESSED_PKL_PATH = os.path.join(DATA_DIR, "ml_ready_features.pkl")

class DLFeatureEngineer:
    def __init__(self, raw_path: str, output_path: str):
        self.raw_path = raw_path
        self.output_path = output_path

    def load_data(self) -> pd.DataFrame:
        print(f"[*] Reading raw data from {self.raw_path}...")
        df = pd.read_csv(self.raw_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[*] Calculating technical indicators via pandas-ta...")
        
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.adx(length=14, append=True)

        df['dist_ema9'] = (df['close'] - df['EMA_9']) / df['EMA_9']
        df['dist_ema21'] = (df['close'] - df['EMA_21']) / df['EMA_21']
        df['dist_ema50'] = (df['close'] - df['EMA_50']) / df['EMA_50']
        df['dist_ema200'] = (df['close'] - df['EMA_200']) / df['EMA_200']

        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)

        df['macd_hist_momentum'] = df['MACDh_12_26_9'] - df['MACDh_12_26_9'].shift(1)

        df.ta.atr(length=14, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        
        df['atr_normalized'] = df['ATRr_14'] / df['close']
         
        #volume
        df.ta.vwap(append=True)
        df['dist_vwap'] = (df['close'] - df['VWAP_D']) / df['VWAP_D']
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_change'] = df['volume'].pct_change()

        return df

    def create_classification_target(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[*] Generating TWIN classification targets (target_long / target_short)...")
        future_window = 6
        profit_threshold = 0.006  # %0.6 hedef
        
        # Gelecekteki 6 mum içinde görülecek en yüksek ve en düşük fiyatlar
        future_highs = df['high'].rolling(window=future_window).max().shift(-future_window)
        future_lows = df['low'].rolling(window=future_window).min().shift(-future_window)
        
        # LONG Hedefi: Fiyat %0.6 YUKARI gidecek mi?
        long_target_prices = df['close'] * (1 + profit_threshold)
        df['target_long'] = (future_highs >= long_target_prices).astype(np.int8)
        
        # SHORT Hedefi: Fiyat %0.6 AŞAĞI gidecek mi?
        short_target_prices = df['close'] * (1 - profit_threshold)
        df['target_short'] = (future_lows <= short_target_prices).astype(np.int8)
        
        return df

    def clean_and_optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        print("[*] Cleaning data and optimizing memory footprint (float32 / int8)...")
        
        # Geleceği gören shift(-window) işlemleri son N satırda NaN bırakır, onları uçur.
        df.dropna(inplace=True)

        # Modeli yanıltmamak için OHLC verilerini silebiliriz, model artık indikatörlere bakacak.
        cols_to_drop = ['open', 'high', 'low']
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

        start_mem = df.memory_usage().sum() / 1024**2
        
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype(np.float32)
            elif df[col].dtype == 'int64':
                if col in ['target_long', 'target_short']:
                    df[col] = df[col].astype(np.int8) # Hedefler 0 veya 1, en küçük int formatı yeterli
                else:
                    df[col] = df[col].astype(np.int32)

        end_mem = df.memory_usage().sum() / 1024**2
        print(f"[*] Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB")
        
        return df

    def execute_pipeline(self):
        df = self.load_data()
        df = self.calculate_technical_features(df)
        df = self.create_classification_target(df)
        df = self.clean_and_optimize_memory(df)

        print("\nTarget Distributions:")
        print(f"LONG Target:  %{df['target_long'].value_counts(normalize=True).get(1, 0) * 100:.2f}")
        print(f"SHORT Target: %{df['target_short'].value_counts(normalize=True).get(1, 0) * 100:.2f}")
        print(f"\n[*] Saving processed features to {self.output_path}...")
        df.to_pickle(self.output_path)
        print("[+] SUCCESS: Pipeline completed.")

if __name__ == "__main__":
    if not os.path.exists(RAW_CSV_PATH):
        print(f"[-] Error: Raw data file not found at {RAW_CSV_PATH}")
    else:
        pipeline = DLFeatureEngineer(RAW_CSV_PATH, PROCESSED_PKL_PATH)
        pipeline.execute_pipeline()
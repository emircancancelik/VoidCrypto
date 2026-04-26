import pandas as pd
import pandas_ta as ta
import os

# Yollar aynı kalacak (Örnek 15m için)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(BASE_DIR, "data", "raw", "BTC_USDT_15m_2018_to_now.csv")
output_path = os.path.join(BASE_DIR, "data", "processed", "BTC_USDT_15m_FEATURES.csv")

df = pd.read_csv(input_path)
df = df.sort_values(by='timestamp').reset_index(drop=True)

# 1. Ham İndikatörleri Hesapla (Geçici olarak)
df.ta.ema(length=9, append=True)
df.ta.ema(length=21, append=True)
df.ta.ema(length=50, append=True)
df.ta.ema(length=200, append=True)
df.ta.macd(fast=12, slow=26, signal=9, append=True)
df.ta.atr(length=14, append=True)

# 2. STATIONARITY (DURAĞANLIK) DÖNÜŞÜMÜ - Kritik Aşama
# Model ham EMA'yı değil, fiyata olan % uzaklığı görecek
df['dist_ema9'] = ((df['close'] - df['EMA_9']) / df['EMA_9']) * 100
df['dist_ema21'] = ((df['close'] - df['EMA_21']) / df['EMA_21']) * 100
df['dist_ema50'] = ((df['close'] - df['EMA_50']) / df['EMA_50']) * 100
df['dist_ema200'] = ((df['close'] - df['EMA_200']) / df['EMA_200']) * 100

# MACD Histogramı zaten sıfır etrafında salındığı için durağandır, direkt alıyoruz
df['macd_hist'] = df['MACDh_12_26_9'] 

# ATR'yi volatilite yüzdesi (Yüzdesel ATR) olarak veriyoruz
df['atr_pct'] = (df['ATRr_14'] / df['close']) * 100 

# RSI ve OBV
df.ta.rsi(length=14, append=True)
df['rsi'] = df['RSI_14']
df.ta.obv(append=True)
df['obv_roc'] = df['OBV'].pct_change() * 100 # OBV'nin değişim hızı

# 3. Eğitime Girecek Sütunları Temizle (Ham fiyat ve ham EMA'ları siliyoruz)
keep_columns = ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume', 
                'dist_ema9', 'dist_ema21', 'dist_ema50', 'dist_ema200', 
                'macd_hist', 'atr_pct', 'rsi', 'obv_roc']

df = df[keep_columns].dropna()
df.to_csv(output_path, index=False)
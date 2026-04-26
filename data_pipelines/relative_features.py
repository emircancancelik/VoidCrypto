import pandas as pd
import numpy as np
import pandas_ta as ta
import os

# --- 1. DOSYA YOLLARI ---
current_file_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(current_file_path))

def process_relative_features(timeframe):
    input_path = os.path.join(BASE_DIR, "data", "raw", f"BTC_USDT_{timeframe}_2018_to_now.csv")
    output_path = os.path.join(BASE_DIR, "data", "raw", f"BTC_USDT_{timeframe}_FEATURES.csv")
    
    if not os.path.exists(input_path):
        print(f"KRİTİK HATA: {timeframe} ham verisi bulunamadı -> {input_path}")
        return

    print(f"\n[{timeframe}] Verisi okunuyor ve Durağanlaştırılıyor (Stationarity)...")
    df = pd.read_csv(input_path)
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    # 1. Ham İndikatörleri Hesapla
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.obv(append=True)

    # 2. DURAĞANLIK DÖNÜŞÜMÜ (Relative Math)
    print(f"[{timeframe}] Matematiksel dönüşümler uygulanıyor...")
    
    # EMA'ların fiyata yüzdesel uzaklığı (Örn: Fiyat, EMA200'ün %5 üzerinde)
    df['dist_ema9'] = ((df['close'] - df['EMA_9']) / df['EMA_9']) * 100
    df['dist_ema21'] = ((df['close'] - df['EMA_21']) / df['EMA_21']) * 100
    df['dist_ema50'] = ((df['close'] - df['EMA_50']) / df['EMA_50']) * 100
    df['dist_ema200'] = ((df['close'] - df['EMA_200']) / df['EMA_200']) * 100

    # MACD'yi fiyata oranlıyoruz (Büyük fiyatlarda MACD şişmesini engeller)
    df['macd_hist_pct'] = (df['MACDh_12_26_9'] / df['close']) * 100
    df['macd_line_pct'] = (df['MACD_12_26_9'] / df['close']) * 100

    # Yüzdesel Volatilite (ATR'yi fiyata bölüyoruz)
    df['atr_pct'] = (df['ATRr_14'] / df['close']) * 100

    # RSI zaten 0-100 arasına sıkışmıştır, durağandır
    df['rsi'] = df['RSI_14']

    # OBV (Kümülatif Hacim) direkt verilirse zehirlidir, Değişim Hızını (ROC) alıyoruz
    df['obv_roc'] = df['OBV'].pct_change() * 100

    # Hacim Değişim Yüzdesi
    df['volume_roc'] = df['volume'].pct_change() * 100

    # 3. KORUNACAK SÜTUNLAR MİMARİSİ
    # Labeling_engine'in TP/SL hesaplayabilmesi için close, high, low sütunlarını bırakıyoruz.
    # Ham EMA, MACD, OBV sütunlarını tamamen siliyoruz ki model bunlara ulaşıp ezber yapmasın.
    keep_columns = [
        'timestamp', 'datetime', 'close', 'high', 'low', 
        'dist_ema9', 'dist_ema21', 'dist_ema50', 'dist_ema200', 
        'macd_hist_pct', 'macd_line_pct', 'atr_pct', 'rsi', 'obv_roc', 'volume_roc'
    ]

    # --- KRİTİK DÜZELTME: Sonsuzlukları (inf) NaN'a çevir ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = df[keep_columns].dropna()
    df.to_csv(output_path, index=False)
    print(f"BAŞARILI: {timeframe} Relative özellikler kaydedildi -> {output_path}")

# Scripti tek tuşla iki zaman dilimi için de çalıştır
if __name__ == "__main__":
    process_relative_features('15m')
    process_relative_features('4h')
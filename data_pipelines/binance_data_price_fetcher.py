import pandas as pd
import pandas_ta as ta
import os

# 1. Klasör yollarını ayarla
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DEĞİŞEN KISIM: Input dosya adı '4h' oldu
input_path = os.path.join(BASE_DIR, "data", "raw", "BTC_USDT_4h_2018_to_now.csv")

# DEĞİŞEN KISIM: Output dosya adı '4h' oldu
output_path = os.path.join(BASE_DIR, "data", "processed", "BTC_USDT_4h_FEATURES.csv")

# Hedef klasörün varlığından emin ol
os.makedirs(os.path.dirname(output_path), exist_ok=True)

if not os.path.exists(input_path):
    print(f"HATA: Ham 4h verisi bulunamadı! Yol: {input_path}")
else:
    print(f"[{input_path}] okunuyor ve 4 saatlik özellikler ekleniyor...")
    df = pd.read_csv(input_path)
    
    # Zaman sıralaması
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    print("4H Teknik Göstergeler Hesaplanıyor (EMA 9, 21, 50, 200, RSI, MACD, ATR, OBV)...")
    
    # İndikatörler (15m ile aynı seti kullanıyoruz ki model hiyerarşisi tutarlı olsun)
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)

    # 2. NaN (Boş Değer) Temizliği
    df.dropna(inplace=True)
    
    # 3. Sonuçları Kaydet
    df.to_csv(output_path, index=False)
    print(f"BAŞARILI: 4h Feature seti hazır -> {output_path}")
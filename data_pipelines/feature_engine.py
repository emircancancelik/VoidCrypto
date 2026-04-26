import asyncio
import aiohttp
import pandas as pd
import pandas_ta as ta
import json
import logging
import redis.asyncio as redis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [FeatureEngine] - %(message)s')

BINANCE_URL = "https://api.binance.com/api/v3/klines"
REDIS_HOST = "localhost"

async def fetch_klines(session, symbol, interval, limit=250):
    """Binance'ten asenkron olarak mum verilerini çeker."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    async with session.get(BINANCE_URL, params=params) as response:
        data = await response.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

async def calculate_and_push():
    r = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)
    
    # aiohttp oturumunu sürekli açık tutuyoruz (Bağlantı gecikmesini engellemek için)
    async with aiohttp.ClientSession() as session:
        logging.info("Feature Engine başlatıldı. Redis'e sürekli özellik pompalanıyor...")
        while True:
            try:
                df = await fetch_klines(session, "BTCUSDT", "15m", limit=250)

                # --- 1. HAREKETLİ ORTALAMALAR VE UZAKLIKLAR ---
                df.ta.ema(length=9, append=True)
                df.ta.ema(length=21, append=True)
                df.ta.ema(length=50, append=True)
                df.ta.ema(length=200, append=True)

                df['dist_ema9'] = (df['close'] - df['EMA_9']) / df['EMA_9']
                df['dist_ema21'] = (df['close'] - df['EMA_21']) / df['EMA_21']
                df['dist_ema50'] = (df['close'] - df['EMA_50']) / df['EMA_50']
                df['dist_ema200'] = (df['close'] - df['EMA_200']) / df['EMA_200']

                # --- 2. MACD VE YÜZDELERİ ---
                df.ta.macd(fast=12, slow=26, signal=9, append=True)
                df['macd_line_pct'] = df['MACD_12_26_9'] / df['close']
                df['macd_hist_pct'] = df['MACDh_12_26_9'] / df['close']

                # --- 3. VOLATİLİTE VE MOMENTUM ---
                df.ta.atr(length=14, append=True)
                df['atr_pct'] = df['ATRr_14'] / df['close']

                df.ta.rsi(length=14, append=True)
                df['rsi'] = df['RSI_14']

                df.ta.obv(append=True)
                df['obv_roc'] = df['OBV'].pct_change()
                df['volume_roc'] = df['volume'].pct_change()

                # --- 4. EN GÜNCEL SATIRI (CANLI MUM) ÇEK ---
                last_row = df.iloc[-1]

                # XGBoost'un midesinin bulandığı o isim hatasını burada kökünden çözüyoruz
                payload = {
                    'dist_ema9': float(last_row['dist_ema9']),
                    'dist_ema21': float(last_row['dist_ema21']),
                    'dist_ema50': float(last_row['dist_ema50']),
                    'dist_ema200': float(last_row['dist_ema200']),
                    'macd_hist_pct': float(last_row['macd_hist_pct']),
                    'macd_line_pct': float(last_row['macd_line_pct']),
                    'atr_pct': float(last_row['atr_pct']),
                    'rsi': float(last_row['rsi']),
                    'obv_roc': float(last_row['obv_roc']),
                    'volume_roc': float(last_row['volume_roc'])
                }

                # --- 5. REDIS'E YAZ ---
                # Ajanların uyandığında bakacağı adres burası
                await r.set("model_features:btcusdt:15m", json.dumps(payload))
                
                # Saniyede bir güncelle (Çok yormadan asenkron bekleme)
                await asyncio.sleep(2) 

            except Exception as e:
                logging.error(f"Veri işleme hatası: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(calculate_and_push())
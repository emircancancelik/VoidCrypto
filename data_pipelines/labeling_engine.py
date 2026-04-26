import pandas as pd
import numpy as np
import os
import time

# --- 1. AYARLAR VE YOLLAR ---
current_file_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(current_file_path))

ATR_MULT = 1.5   
RR_RATIO = 1.5   
HORIZON = 48     

def apply_volatility_labels(df, timeframe):
    print(f"\n[{timeframe}] ATR-Tabanlı Etiketleme Başladı...")
    start_time = time.time()
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atr_values = df['atr_pct'].values / 100 
    
    n = len(df)
    target_long = np.zeros(n, dtype=int)
    target_short = np.zeros(n, dtype=int)
    
    for i in range(n - 1):
        entry_price = closes[i]
        curr_atr = atr_values[i]
        
        sl_dist = curr_atr * ATR_MULT
        tp_dist = sl_dist * RR_RATIO
        
        l_tp, l_sl = entry_price * (1 + tp_dist), entry_price * (1 - sl_dist)
        s_tp, s_sl = entry_price * (1 - tp_dist), entry_price * (1 + sl_dist)
        
        limit = min(i + 1 + HORIZON, n)
        
        for j in range(i + 1, limit):
            if lows[j] <= l_sl: break 
            if highs[j] >= l_tp: 
                target_long[i] = 1
                break
                
        for j in range(i + 1, limit):
            if highs[j] >= s_sl: break 
            if lows[j] <= s_tp: 
                target_short[i] = 1
                break

    df['target_long'] = target_long
    df['target_short'] = target_short
    
    print(f"[{timeframe}] Etiketleme Bitti. Süre: {time.time() - start_time:.2f}s")
    print(f"[{timeframe}] Karlı Oranlar -> L: %{df.target_long.mean()*100:.2f} | S: %{df.target_short.mean()*100:.2f}")
    return df

def process_and_save(timeframe):
    input_path = os.path.join(BASE_DIR, "data", "raw", f"BTC_USDT_{timeframe}_FEATURES.csv")
    output_path = os.path.join(BASE_DIR, "data", "raw", f"BTC_USDT_{timeframe}_LABELED.csv")
    
    if not os.path.exists(input_path):
        print(f"HATA: {input_path} bulunamadı.")
        return
        
    df = pd.read_csv(input_path)
    # FONKSİYON İSMİ DÜZELTİLDİ:
    df = apply_volatility_labels(df, timeframe)
    
    drop_cols = ['open', 'high', 'low', 'close']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')
    
    df.to_csv(output_path, index=False)
    print(f"[{timeframe}] Dosya Kaydedildi -> {output_path}")

if __name__ == "__main__":
    process_and_save('15m')
    process_and_save('4h')
import pandas as pd
import xgboost as xgb
import os

# --- 1. AYARLAR VE YOLLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_15m_path = os.path.join(BASE_DIR, "data", "raw", "BTC_USDT_15m_FEATURES.csv")
data_4h_path = os.path.join(BASE_DIR, "data", "raw", "BTC_USDT_4h_FEATURES.csv")

# Labeling engine labelli dosyaları ürettiği için yolları LABELED olarak düzeltiyoruz
data_15m_path = data_15m_path.replace("FEATURES", "LABELED")
data_4h_path = data_4h_path.replace("FEATURES", "LABELED")

model_15m_path = os.path.join(BASE_DIR, "agents", "void_model_15m.json")
model_4h_path = os.path.join(BASE_DIR, "agents", "void_model_4h.json")

for p in [data_15m_path, data_4h_path, model_15m_path, model_4h_path]:
    if not os.path.exists(p):
        print(f"KRİTİK HATA: Dosya bulunamadı -> {p}")
        exit()

# --- 2. CÜZDAN AYARLARI ---
INITIAL_CAPITAL = 1000.0
TRADE_PERCENT = 0.10      
COMMISSION_RATE = 0.001   
TP_RATE = 0.0115          
SL_RATE = 0.0100          

CONFIDENCE_15M = 0.65  # %85'ten 65'e çektik (Durağan veri için çok iyi bir oran)
CONFIDENCE_4H = 0.55   # Makro trendin yönünü teyit etmesi yeterli

print("Veriler ve Modeller Yükleniyor...")
df_15m = pd.read_csv(data_15m_path)
df_4h = pd.read_csv(data_4h_path)

model_15m = xgb.XGBClassifier()
model_15m.load_model(model_15m_path)

model_4h = xgb.XGBClassifier()
model_4h.load_model(model_4h_path)

# --- 3. 4H TAHMİNLERİNİ HESAPLA ---
drop_cols = ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'target']
X_4h = df_4h.drop(columns=drop_cols, errors='ignore')
df_4h['prob_4h'] = model_4h.predict_proba(X_4h)[:, 1]

# --- DÜZELTME 2: Deterministik Filtre İçin EMA 200 Uzaklığını Al ---
df_4h['datetime'] = pd.to_datetime(df_4h['datetime'])
df_macro = df_4h[['datetime', 'prob_4h', 'dist_ema200']].copy()
df_macro.rename(columns={'dist_ema200': 'macro_ema200'}, inplace=True)

# --- 4. 15M TAHMİNLERİNİ HESAPLA VE BİRLEŞTİR ---
df_15m['datetime'] = pd.to_datetime(df_15m['datetime'])
X_15m = df_15m.drop(columns=drop_cols, errors='ignore')
df_15m['prob_15m'] = model_15m.predict_proba(X_15m)[:, 1]

print("Ajanlar konsensüs için birleştiriliyor...")
df_15m = df_15m.sort_values('datetime')
df_macro = df_macro.sort_values('datetime')
merged_df = pd.merge_asof(df_15m, df_macro, on='datetime', direction='backward')

train_size = int(len(merged_df) * 0.8)
test_df = merged_df.iloc[train_size:].copy()

# --- 5. MASTER AI BACKTEST DÖNGÜSÜ ---
capital = INITIAL_CAPITAL
wins = 0
losses = 0
total_commissions = 0.0

print("\n--- MASTER ORCHESTRATOR BACKTEST BAŞLIYOR ---")
for index, row in test_df.iterrows():
    
    # --- DÜZELTME 3: DİKTATÖR KURALI ---
    # 1. 15m Yüksek Olasılık VE
    # 2. 4H Yüksek Olasılık VE
    # 3. Deterministik Kural: 4 Saatlik fiyat 200 EMA'nın üzerinde olmalı (Değer > 0)
    
    if row['prob_15m'] >= CONFIDENCE_15M and row['prob_4h'] >= CONFIDENCE_4H and row['macro_ema200'] > 0:
        
        position_size = capital * TRADE_PERCENT
        entry_fee = position_size * COMMISSION_RATE
        total_commissions += entry_fee
        
        if row['target'] == 1: 
            exit_fee = (position_size * (1 + TP_RATE)) * COMMISSION_RATE
            net_profit = (position_size * TP_RATE) - entry_fee - exit_fee
            capital += net_profit
            wins += 1
            total_commissions += exit_fee
        else: 
            exit_fee = (position_size * (1 - SL_RATE)) * COMMISSION_RATE
            net_loss = (position_size * SL_RATE) + entry_fee + exit_fee
            capital -= net_loss
            losses += 1
            total_commissions += exit_fee

# --- SONUÇLAR ---
total_trades = wins + losses
win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
roi = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

print("\n" + "="*30)
print(f"BİTİŞ KASASI    : ${capital:.2f}")
print(f"NET GETİRİ (ROI): %{roi:.2f}")
print("="*30)
print(f"Toplam İşlem Sayısı: {total_trades}")
print(f"Başarılı (Win)     : {wins}")
print(f"Başarısız (Loss)   : {losses}")
print(f"Win Rate           : %{win_rate:.2f}")
print(f"Ödenen Komisyon    : ${total_commissions:.2f}")
print("="*30)
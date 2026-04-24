import pandas as pd
import xgboost as xgb
import os

# --- 1. AYARLAR VE YOLLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "raw", "BTC_USDT_15m_LABELED.csv")
model_path = os.path.join(BASE_DIR, "agents", "void_model_15m.json")

# --- 2. CÜZDAN VE İŞLEM MATEMATİĞİ ---
INITIAL_CAPITAL = 1000.0  # Başlangıç bakiyesi (Dolar)
TRADE_PERCENT = 0.10      # Her işlemde kasanın %10'u ile girilecek (Risk Yönetimi)
COMMISSION_RATE = 0.001   # %0.1 Binance komisyonu
TP_RATE = 0.0115          # %1.15 Kar Al (Labeling'de belirlediğimiz oran)
SL_RATE = 0.0100          # %1.00 Zarar Durdur
CONFIDENCE_THRESHOLD = 0.65 # Model "Ben %65 eminim" demezse işleme girme!

print("Veri ve Model Yükleniyor...")
if not os.path.exists(data_path):
    print(f"\nKRİTİK HATA: Veri dosyası bulunamadı!")
    print(f"Aranan yol: {data_path}")
    exit()

if not os.path.exists(model_path):
    print(f"\nKRİTİK HATA: Model dosyası (.json) bulunamadı!")
    print(f"Aranan yol: {model_path}")
    print("ÇÖZÜM: 'train_model.py' dosyasını çalıştırıp en sonda 'model.save_model()' komutunun çalıştığından emin ol.")
    exit()


df = pd.read_csv(data_path)
model = xgb.XGBClassifier()
model.load_model(model_path)

# Sadece test verisinde (modelin eğitimde görmediği son %20'lik kısım) test edelim
train_size = int(len(df) * 0.8)
test_df = df.iloc[train_size:].copy()

drop_columns = ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'target']
X_test = test_df.drop(columns=drop_columns)

# Modelin 1 (Başarılı) olma ihtimalini yüzdelik (probability) olarak al
print("Model Geleceği Tahmin Ediyor...")
probabilities = model.predict_proba(X_test)[:, 1]
test_df['predicted_prob'] = probabilities

# --- 3. BACKTEST DÖNGÜSÜ ---
capital = INITIAL_CAPITAL
wins = 0
losses = 0
total_commissions_paid = 0.0

print("\n--- BACKTEST BAŞLIYOR ---")
for index, row in test_df.iterrows():
    # Sinyal güçlü mü?
    if row['predicted_prob'] >= CONFIDENCE_THRESHOLD:
        
        position_size = capital * TRADE_PERCENT
        
        # Giriş komisyonu
        entry_fee = position_size * COMMISSION_RATE
        total_commissions_paid += entry_fee
        
        # Gerçekte (veride) bu işlem başarılı olmuş mu? (Labeling: 1 ise TP, 0 ise SL vurmuştur)
        if row['target'] == 1:
            # TP Vurursa: Pozisyon * Kâr Oranı - Çıkış Komisyonu
            exit_fee = (position_size * (1 + TP_RATE)) * COMMISSION_RATE
            net_profit = (position_size * TP_RATE) - entry_fee - exit_fee
            
            capital += net_profit
            wins += 1
            total_commissions_paid += exit_fee
        else:
            # SL Vurursa: Pozisyon * Zarar Oranı + Çıkış Komisyonu
            exit_fee = (position_size * (1 - SL_RATE)) * COMMISSION_RATE
            net_loss = (position_size * SL_RATE) + entry_fee + exit_fee
            
            capital -= net_loss
            losses += 1
            total_commissions_paid += exit_fee

# --- 4. SONUÇ RAPORU ---
win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
roi = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

print("\n" + "="*30)
print(f"BAŞLANGIÇ KASASI: ${INITIAL_CAPITAL:.2f}")
print(f"BİTİŞ KASASI    : ${capital:.2f}")
print(f"NET GETİRİ (ROI): %{roi:.2f}")
print("="*30)
print(f"Toplam İşlem Sayısı: {wins + losses}")
print(f"Başarılı (Win)     : {wins}")
print(f"Başarısız (Loss)   : {losses}")
print(f"Win Rate           : %{win_rate:.2f}")
print(f"Ödenen Komisyon    : ${total_commissions_paid:.2f} (Borsaya Giden)")
print("="*30)
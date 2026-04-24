import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# 1. DOSYA YOLLARINI KESİNLEŞTİR
current_file_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(current_file_path))
data_path = os.path.join(BASE_DIR, "data", "raw", "BTC_USDT_15m_LABELED.csv")

print(f"Veri aranıyor: {data_path}")

if not os.path.exists(data_path):
    print("HATA: Dosya bulunamadı! Lütfen yolu kontrol et.")
    exit(1)

# 2. VERİYİ YÜKLE
df = pd.read_csv(data_path)
print(f"Veri başarıyla yüklendi. Satır sayısı: {len(df)}")

# Eğer eski formatta 'target' varsa ve twin hedefler yoksa uyar
if 'target_long' not in df.columns or 'target_short' not in df.columns:
    print("KRİTİK HATA: Veri setinde 'target_long' ve 'target_short' kolonları bulunamadı!")
    print("Veri etiketleme (labeling) aşamasında tek bir 'target' yerine ikiz hedefler oluşturmalısın.")
    exit(1)

# 3. ÖZELLİKLERİ (FEATURES) AYIR
# Modellerin geleceği görmesini (data leakage) engellemek için hedefleri siliyoruz
drop_columns = ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'target', 'target_long', 'target_short']
X = df.drop(columns=[col for col in drop_columns if col in df.columns])

# İkiz Hedefler
y_long = df['target_long']
y_short = df['target_short']

# 4. ZAMAN SERİSİ BÖLÜMLEME (Zamanın akışını bozmamak için shuffle=False)
X_train, X_test, y_train_long, y_test_long = train_test_split(X, y_long, test_size=0.2, random_state=42, shuffle=False)
_, _, y_train_short, y_test_short = train_test_split(X, y_short, test_size=0.2, random_state=42, shuffle=False)

# 5. HARD-LOGIC: IMBALANCE (Dengesizlik) HESAPLAMALARI
print("\n--- Sınıf Dengesizliği (Class Imbalance) Analizi ---")
# LONG Modeli için Oran
long_negatives = (y_train_long == 0).sum()
long_positives = (y_train_long == 1).sum()
spw_long = float(long_negatives / long_positives) if long_positives > 0 else 1.0
print(f"LONG Ajanı -> Negatif: {long_negatives}, Pozitif: {long_positives} | Atanan spw_long: {spw_long:.3f}")

# SHORT Modeli için Oran
short_negatives = (y_train_short == 0).sum()
short_positives = (y_train_short == 1).sum()
spw_short = float(short_negatives / short_positives) if short_positives > 0 else 1.0
print(f"SHORT Ajanı -> Negatif: {short_negatives}, Pozitif: {short_positives} | Atanan spw_short: {spw_short:.3f}")

# 6. MODELLERİ YAPILANDIR VE EĞİT
print("\n--- XGBoost LONG Ajanı Eğitiliyor ---")
model_long = xgb.XGBClassifier(
    n_estimators=200,             
    max_depth=5,                  # 8 çok derindi, Overfitting'i önlemek için 5'e çektik
    learning_rate=0.05,           
    scale_pos_weight=spw_long,    # LONG ağırlığı
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    random_state=42
)
model_long.fit(X_train, y_train_long)

print("\n--- XGBoost SHORT Ajanı Eğitiliyor ---")
model_short = xgb.XGBClassifier(
    n_estimators=200,             
    max_depth=5,                  
    learning_rate=0.05,           
    scale_pos_weight=spw_short,   # SHORT ağırlığı (Sistemi kurtaracak olan kısım burası)
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    random_state=42
)
model_short.fit(X_train, y_train_short)

# 7. SONUÇLARI RAPORLA
print("\n" + "="*40)
print("LONG MODEL PERFORMANSI (Test Seti)")
print("="*40)
y_pred_long = model_long.predict(X_test)
print(classification_report(y_test_long, y_pred_long))

print("\n" + "="*40)
print("SHORT MODEL PERFORMANSI (Test Seti)")
print("="*40)
y_pred_short = model_short.predict(X_test)
print(classification_report(y_test_short, y_pred_short))

# 8. MODELLERİ KAYDET
model_long_path = os.path.join(BASE_DIR, "agents", "void_model_15m_long.json")
model_short_path = os.path.join(BASE_DIR, "agents", "void_model_15m_short.json")

os.makedirs(os.path.dirname(model_long_path), exist_ok=True)

model_long.save_model(model_long_path)
model_short.save_model(model_short_path)

print("\nBAŞARILI: İkiz (Twin) modeller yaratıldı ve kaydedildi:")
print(f"-> {model_long_path}")
print(f"-> {model_short_path}")
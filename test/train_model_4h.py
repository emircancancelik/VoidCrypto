import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os

# 1. DOSYA YOLLARINI KESİNLEŞTİR
current_file_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(current_file_path))

# DÜZELTME 1: 4H Verisi Okunuyor
data_path = os.path.join(BASE_DIR, "data", "raw", "BTC_USDT_4h_LABELED.csv")

print(f"Veri aranıyor: {data_path}")

if not os.path.exists(data_path):
    print(f"HATA: Dosya bulunamadı! Lütfen yolu kontrol et.")
else:
    # 2. VERİYİ YÜKLE
    df = pd.read_csv(data_path)
    print(f"Veri başarıyla yüklendi. Satır sayısı: {len(df)}")

    # 3. ÖZELLİKLERİ AYIR
    drop_columns = ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'target']
    X = df.drop(columns=drop_columns, errors='ignore')
    y = df['target']

    # 4. ZAMAN SERİSİ BÖLÜMLEME
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # 5. MODELİ YAPILANDIR VE EĞİT
    ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])
    # DÜZELTME: adjusted_ratio çarpanını kaldırdık. Direkt doğal oranı kullanıyoruz.

    model = xgb.XGBClassifier(
        n_estimators=300,             # Ağaç sayısını normalleştirdik
        max_depth=3,                  # Derinliği 3'e düşürdük (Sadece ana trendi görsün)
        learning_rate=0.05,           
        min_child_weight=5,           # Gürültüye karşı çok daha katı filtre
        scale_pos_weight=ratio,       # Sadece doğal dengesizliği telafi et
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42
    )
    
    print("\n4H XGBoost Eğitimi Başlıyor...")
    model.fit(X_train, y_train)

    # 6. SONUÇLARI RAPORLA
    y_pred = model.predict(X_test)
    print("\n" + "="*30)
    print(f"4H Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print("="*30)
    print(classification_report(y_test, y_pred))

    # --- DÜZELTME 2 ve 3: ÖNCE KLASÖRÜ GARANTİLE, SONRA KAYDET, EN SON GRAFİĞİ ÇİZ ---
    
    # 7. MODELİ KAYDET
    model_save_path = os.path.join(BASE_DIR, "agents", "void_model_4h.json")
    
    # agents klasörünün varlığından emin ol, yoksa oluştur
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # JSON Dosyasını diske yaz
    model.save_model(model_save_path)
    print(f"\nBAŞARILI: 4H Model dosyası yaratıldı ve kaydedildi -> {model_save_path}")

    # 8. ÖNEMLİ İNDİKATÖRLERİ GÖRSELLEŞTİR
    print("Grafik açılıyor... Grafiği kapattığınızda işlem sonlanacaktır.")
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, ax=ax, max_num_features=12, importance_type='weight')
    plt.show()
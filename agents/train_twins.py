import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# --- 1. AYARLAR VE YOLLAR ---
current_file_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(current_file_path))

def train_agent(timeframe, direction):
    """
    Belirli bir zaman dilimi (15m, 4h) ve yön (long, short) için XGBoost modeli eğitir.
    """
    data_path = os.path.join(BASE_DIR, "data", "raw", f"BTC_USDT_{timeframe}_LABELED.csv")
    model_save_path = os.path.join(BASE_DIR, "agents", f"void_model_{timeframe}_{direction}.json")
    
    print(f"\n{'='*50}")
    print(f"EĞİTİM BAŞLIYOR: [{timeframe}] - {direction.upper()} Uzmanı")
    print(f"{'='*50}")

    if not os.path.exists(data_path):
        print(f"HATA: Dosya bulunamadı -> {data_path}")
        return

    # Veriyi Yükle
    df = pd.read_csv(data_path)
    
    # 1. Özellikleri Ayır
    drop_columns = ['timestamp', 'datetime', 'target_long', 'target_short']
    X = df.drop(columns=drop_columns, errors='ignore')
    
    # Hangi yöne (target_long veya target_short) odaklanıyoruz?
    y_col = f"target_{direction}"
    y = df[y_col]

    # 2. Zaman Serisi Bölümleme (Geçmişle geleceği eğit, %80 Train, %20 Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # 3. Model Parametreleri (Durağan veriye göre kalibre edilmiş)
    # 4H için biraz daha az derinlik kullanacağız
    max_depth = 4 if timeframe == '15m' else 3
    n_estimators = 500 if timeframe == '15m' else 300
    
    # Oranları hesapla ve çok hafif bir destek ver
    ratio = float(y_train.value_counts()[0] / y_train.value_counts()[1])

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.01 if timeframe == '15m' else 0.05,
        min_child_weight=3 if timeframe == '15m' else 5,
        scale_pos_weight=ratio, 
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42
    )

    # 4. Modeli Eğit
    print("XGBoost beyni inşa ediliyor...")
    model.fit(X_train, y_train)

    # 5. Sonuçları Raporla
    y_pred = model.predict(X_test)
    print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # 6. Modeli Kaydet
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save_model(model_save_path)
    print(f"BAŞARILI: Model kaydedildi -> {model_save_path}")

if __name__ == "__main__":
    # Sırayla 4 uzmanı (İkiz Ajanlar) eğitiyoruz
    train_agent('15m', 'long')
    train_agent('15m', 'short')
    train_agent('4h', 'long')
    train_agent('4h', 'short')
    
    print("\nTÜM AJANLARIN EĞİTİMİ VE KAYDI TAMAMLANDI.")
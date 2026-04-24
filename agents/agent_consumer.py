import asyncio
import json
import logging
import aio_pika
import xgboost as xgb
import pandas as pd
import os
from pathlib import Path
import redis.asyncio as redis

# --- LOGLAMA ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MasterAI] - %(message)s')
logger = logging.getLogger(__name__)

# --- YOLLAR VE AYARLAR ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_LONG_PATH = os.path.join(BASE_DIR, "agents", "void_model_15m_long.json")
MODEL_SHORT_PATH = os.path.join(BASE_DIR, "agents", "void_model_15m_short.json")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost/")
CONFIDENCE_THRESHOLD = 0.60  # %60 altı tahminleri işleme sokma

class MasterOrchestrator:
    def __init__(self):
        self.model_long = None
        self.model_short = None
        self._load_models()

    def _load_models(self):
        """Konteyner ayağa kalktığında İkiz Modelleri RAM'e yükler (Soğuk Başlangıç)."""
        logger.info("İkiz Yapay Zeka Modelleri (Twin Agents) belleğe yükleniyor...")
        if not os.path.exists(MODEL_LONG_PATH) or not os.path.exists(MODEL_SHORT_PATH):
            logger.error("KRİTİK HATA: Model JSON dosyaları bulunamadı!")
            exit(1)
            
        self.model_long = xgb.XGBClassifier()
        self.model_long.load_model(MODEL_LONG_PATH)
        
        self.model_short = xgb.XGBClassifier()
        self.model_short.load_model(MODEL_SHORT_PATH)
        logger.info("Modeller operasyona hazır.")

    def __init__(self):
        self.model_long = None
        self.model_short = None
        # Consumer için Asenkron Redis bağlantısını başlat
        self.redis = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        self._load_models()

    async def get_realtime_features(self, symbol: str) -> pd.DataFrame:
        """
        API bekleme süresi SIFIR. 
        KEDA uyandırdığı an doğrudan RAM'den (Redis) veriyi çeker.
        """
        raw_data = await self.redis.get(f"model_features:{symbol.lower()}:15m")
        if not raw_data:
            raise ValueError("Redis'te hesaplanmış özellik (feature) bulunamadı! Feature Engine çalışıyor mu?")
            
        data = json.loads(raw_data)
        df = pd.DataFrame([data])
        
        # Sütun sırasının XGBoost için kusursuz olduğundan emin oluyoruz
        expected_cols = ['dist_ema9', 'dist_ema21', 'dist_ema50', 'dist_ema200', 'macd_hist_pct', 'macd_line_pct', 'atr_pct', 'rsi', 'obv_roc', 'volume_roc']
        return df[expected_cols]

    async def execute_consensus(self, payload: dict):
        """Ajanların uyandığı ve karar verdiği ana metot."""
        symbol = payload.get("symbol")
        price = payload.get("price")
        
        logger.info(f"⚡ TETİKLEYİCİ: {symbol} anlık fırlama yaptı! (Fiyat: ${price})")
        logger.info("TA ve DL Ajanları güncel piyasa bağlamını (Market Context) analiz ediyor...")
        
        # 1. Güncel veriyi çek ve XGBoost'un anlayacağı formata (DataFrame) sok
        features_df = await self.get_realtime_features(symbol)        
        # 2. LONG Ajanı Fikri
        prob_long = self.model_long.predict_proba(features_df)[0][1]
        
        # 3. SHORT Ajanı Fikri
        prob_short = self.model_short.predict_proba(features_df)[0][1]
        
        logger.info(f"[KARAR AŞAMASI] LONG Olasılığı: %{prob_long*100:.1f} | SHORT Olasılığı: %{prob_short*100:.1f}")
        
        # 4. Deterministik Master Kararı (Konsensüs)
        if prob_long >= CONFIDENCE_THRESHOLD and prob_long > prob_short:
            logger.warning(f"🚀 KONSENSÜS SAĞLANDI: YÖN [LONG]. (Risk/Execution AI'a Emir Gönderiliyor)")
        elif prob_short >= CONFIDENCE_THRESHOLD and prob_short > prob_long:
            logger.warning(f"💥 KONSENSÜS SAĞLANDI: YÖN [SHORT]. (Risk/Execution AI'a Emir Gönderiliyor)")
        else:
            logger.info("⚖️ KARAR: BEKLE. (Güven skoru yeterli değil veya piyasa kararsız).")
        
        print("-" * 60)

async def process_message(message: aio_pika.IncomingMessage, orchestrator: MasterOrchestrator):
    """RabbitMQ'dan mesajı güvenle çeker, işler ve Acknowledge (Onay) gönderir."""
    async with message.process(): 
        payload = json.loads(message.body.decode())
        await orchestrator.execute_consensus(payload)

async def main():
    orchestrator = MasterOrchestrator()
    try:
        connection = await aio_pika.connect_robust(RABBITMQ_URL)
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=1) # Aynı anda sadece 1 sinyal işle (Determinizm)
            
            queue = await channel.declare_queue("wakeup_queue", durable=True)
            logger.info("Master AI RabbitMQ Tüketicisi (Consumer) Başlatıldı. Emir bekleniyor...")
            
            # Parametre ile metot çağırmak için lambda/wrapper kullanıyoruz
            await queue.consume(lambda msg: process_message(msg, orchestrator))
            await asyncio.Future() # Sonsuz döngü
            
    except Exception as e:
        logger.error(f"Sistem Başlatılamadı: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Sistem kapatıldı.")
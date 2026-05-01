import asyncio
import json
import logging
import os
import signal
import time
from typing import Optional

import pandas as pd
import redis.asyncio as redis
import xgboost as xgb
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "agent": "MLPriceActionAI", "level": "%(levelname)s", "message": %(message)s}',
)
logger = logging.getLogger("MLPriceActionAI")


class Config:
    REDIS_URL = os.getenv("REDIS_URL", "redis://redis-service:6379/0")
    # KEDA'nın izlediği kuyruk
    REQUEST_QUEUE_KEY = os.getenv("REQUEST_QUEUE_KEY", "ml_price_action_requests")
    # Ajanın boşta bekleme süresi (KEDA minReplica: 0 için)
    IDLE_SHUTDOWN_SEC = int(os.getenv("IDLE_SHUTDOWN_SEC", "30"))
    
    # Model Yolları
    MODEL_LONG_PATH = os.getenv("MODEL_LONG_PATH", "/app/models/ml_action_long.json")
    MODEL_SHORT_PATH = os.getenv("MODEL_SHORT_PATH", "/app/models/ml_action_short.json")

    EXPECTED_FEATURES = [
        "dist_ema9", "dist_ema21", "dist_ema50", "dist_ema200",
        "macd_hist_pct", "macd_line_pct", "atr_pct", "rsi", "obv_roc", "volume_roc"
    ]



class PriceActionRequest(BaseModel):
    symbol: str
    request_id: str
    timestamp: float = Field(default_factory=time.time)


class MLPriceActionAI:
    AGENT_ID = "MLPriceActionAI"

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.model_long: Optional[xgb.XGBClassifier] = None
        self.model_short: Optional[xgb.XGBClassifier] = None
        self._shutdown_event = asyncio.Event()

    async def connect(self):
        self.redis_client = await redis.from_url(Config.REDIS_URL, decode_responses=True)
        logger.info('"Redis bağlantısı kuruldu."')

    def load_models(self):
        logger.info('"Modeller yükleniyor..."')
        try:
            self.model_long = xgb.XGBClassifier()
            self.model_short = xgb.XGBClassifier()
            
            # Geliştirme aşamasında mock tahmin yapmak için model dosyalarını kontrol et
            if os.path.exists(Config.MODEL_LONG_PATH):
                self.model_long.load_model(Config.MODEL_LONG_PATH)
            
            if os.path.exists(Config.MODEL_SHORT_PATH):
                self.model_short.load_model(Config.MODEL_SHORT_PATH)
                
            logger.info('"Modeller yüklendi veya mock moda geçildi."')
        except Exception as e:
            logger.error(f'"Model yükleme hatası: {e}"')
            raise SystemExit(1)

    async def _get_features(self, symbol: str) -> Optional[pd.DataFrame]:
        # TODO: Gerçek sistemde FeatureEngine'in güncellediği anahtar kullanılmalı
        key = f"model_features:{symbol.lower()}:15m"
        try:
            raw_data = await self.redis_client.get(key)
            if not raw_data:
                return None
            
            data = json.loads(raw_data)
            df = pd.DataFrame([data])
            
            # Eksik kolonları 0 ile doldur
            for col in Config.EXPECTED_FEATURES:
                if col not in df.columns:
                    df[col] = 0.0
                    
            return df[Config.EXPECTED_FEATURES]
        except Exception as e:
            logger.error(f'"Özellik çekme hatası: {e}"')
            return None

    def _infer(self, df: pd.DataFrame) -> tuple[str, float]:
        """XGBoost inference yapar. Model yoksa mock döner."""
        try:
            if not os.path.exists(Config.MODEL_LONG_PATH):
                return "PASS", 0.0
                
            prob_long = float(self.model_long.predict_proba(df)[0][1])
            prob_short = float(self.model_short.predict_proba(df)[0][1])

            if prob_long > prob_short and prob_long > 0.5:
                return "LONG", prob_long
            elif prob_short > prob_long and prob_short > 0.5:
                return "SHORT", prob_short
            else:
                return "PASS", max(prob_long, prob_short)
        except Exception as e:
            logger.error(f'"Inference hatası: {e}"')
            return "PASS", 0.0

    async def _process_request(self, raw_req: str):
        try:
            req = PriceActionRequest.model_validate_json(raw_req)
        except Exception as e:
            logger.error(f'"İstek ayrıştırma hatası: {e}"')
            return

        features_df = await self._get_features(req.symbol)
        
        if features_df is None:
            logger.warning(f'"[{req.symbol}] İçin özellik (feature) bulunamadı. PASS dönülüyor."')
            direction, confidence = "PASS", 0.0
        else:
            direction, confidence = self._infer(features_df)
        redis_key = f"agent_signal:ml_price_ai:{req.symbol.lower()}"
        payload = {
            "direction": direction,
            "confidence": round(confidence, 4),
            "timestamp": time.time()
        }
        await self.redis_client.setex(redis_key, 60, json.dumps(payload))
        
        logger.info(json.dumps({
            "event": "INFERENCE_COMPLETE",
            "symbol": req.symbol,
            "direction": direction,
            "confidence": payload["confidence"]
        }))

    async def serve(self):
        await self.connect()
        self.load_models()
        logger.info('"ML Price Action AI serve modunda. Kuyruk bekleniyor."')

        idle_seconds = 0

        try:
            while not self._shutdown_event.is_set():
                result = await self.redis_client.brpop(
                    Config.REQUEST_QUEUE_KEY,
                    timeout=Config.IDLE_SHUTDOWN_SEC,
                )

                if result is None:
                    idle_seconds += Config.IDLE_SHUTDOWN_SEC
                    logger.info(f'"Idle: {idle_seconds}s. Kapatılıyor."')
                    break

                _, raw_req = result
                idle_seconds = 0
                await self._process_request(raw_req)

        finally:
            if self.redis_client:
                await self.redis_client.aclose()
            logger.info('"Agent kapatıldı."')

    def _handle_shutdown(self, signum):
        logger.info('"Kapatma sinyali alındı."')
        self._shutdown_event.set()

    async def run(self):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown, sig)
        
        await self.serve()

if __name__ == "__main__":
    agent = MLPriceActionAI()
    asyncio.run(agent.run())
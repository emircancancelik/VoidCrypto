import asyncio
import hashlib
import json
import logging
import os
import signal
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import redis.asyncio as redis
import xgboost as xgb
from pydantic import BaseModel, Field, field_validator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import joblib

# ─── JSON Structured Logging (Azure Monitor / Log Analytics compatible) ───────
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "agent": "MLPriceActionAI", "level": "%(levelname)s", "message": %(message)s}',
)
logger = logging.getLogger("MLPriceActionAI")

class ProbaCalibrator:
    def __init__(self, n_classes=2):
        from sklearn.isotonic import IsotonicRegression
        self.n_classes = n_classes
        self.regressors = [IsotonicRegression(out_of_bounds='clip') for _ in range(n_classes)]

    def fit(self, proba, y):
        for i in range(self.n_classes):
            self.regressors[i].fit(proba[:, i], (y == i).astype(int))
    
    def predict_proba(self, proba):
        import numpy as np
        calibrated = np.zeros_like(proba)
        for i in range(self.n_classes):
            calibrated[:, i] = self.regressors[i].transform(proba[:, i])
        # Normalize
        row_sums = calibrated.sum(axis=1, keepdims=True)
        return calibrated / np.where(row_sums == 0, 1, row_sums)
    
# ─── Configuration ────────────────────────────────────────────────────────────
class Config:
    REDIS_URL     = os.getenv("REDIS_URL", "redis://redis-service:6379/0")
    REQUEST_QUEUE = os.getenv("REQUEST_QUEUE_KEY", "ml_price_action_requests")
    RESULT_TTL_SEC    = int(os.getenv("RESULT_TTL_SEC", "30"))
    BRPOP_TIMEOUT_SEC = int(os.getenv("BRPOP_TIMEOUT_SEC", "300"))
    MODEL_PATH        = os.getenv("MODEL_PATH", "agents/void_model_15m_multiclass.json")
    CALIBRATOR_PATH   = os.getenv("CALIBRATOR_PATH", "agents/void_calibrator_15m.pkl")
    MODEL_SHA256      = os.getenv("MODEL_SHA256", "")  
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
    CLASS_MAP = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}
    CLASS_SIGNAL_MAP = {0: -1, 1: 0, 2: 1}

    EXPECTED_FEATURES = [
        "dist_ema9", "dist_ema21", "dist_ema50", "dist_ema200",
        "macd_hist_pct", "macd_line_pct", "atr_pct", "rsi",
        "obv_roc", "volume_roc",
    ]


# ─── Schemas ──────────────────────────────────────────────────────────────────
class PriceActionRequest(BaseModel):
    symbol: str
    request_id: str
    timestamp: float = Field(default_factory=time.time)

    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.upper().strip()


class PriceActionSignal(BaseModel):
    agent: str = "ml_price_ai"
    symbol: str
    request_id: str
    signal: int                  # -1 / 0 / 1
    signal_label: str            # SHORT / NEUTRAL / LONG
    confidence: float            # calibrated P(predicted_class)
    raw_proba: list[float]       # full calibrated probability vector [P_short, P_neutral, P_long]
    predicted_class: int         # raw model class index (0/1/2)
    timestamp: float


# ─── Agent ────────────────────────────────────────────────────────────────────
class MLPriceActionAI:
    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._stop_event = asyncio.Event()
        self._model: Optional[xgb.XGBClassifier] = None
        self._calibrator = None   # sklearn CalibratedClassifierCV or compatible

    # ── Model Loading ──────────────────────────────────────────────────────────
    def _verify_model_integrity(self, path: str) -> bool:
        """SHA-256 hash check. Skipped if MODEL_SHA256 env var is not set."""
        if not Config.MODEL_SHA256:
            logger.warning('"MODEL_SHA256 not set — skipping integrity check."')
            return True
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if sha != Config.MODEL_SHA256:
            logger.error(f'"Model integrity FAILED. Expected={Config.MODEL_SHA256}, Got={sha}"')
            return False
        return True

    def load_models(self) -> None:
        """
        Loads XGBoost multi-class model and its Isotonic/Platt calibrator.
        Hard failure on missing files or integrity mismatch.
        """
        for path in (Config.MODEL_PATH, Config.CALIBRATOR_PATH):
            if not os.path.exists(path):
                logger.critical(f'"Asset missing: {path}. Agent cannot start."')
                raise SystemExit(1)

        if not self._verify_model_integrity(Config.MODEL_PATH):
            raise SystemExit(1)

        self._model = xgb.XGBClassifier()
        self._model.load_model(Config.MODEL_PATH)

        # Calibrator is a pre-fitted sklearn object (saved after calibration training)
        self._calibrator = joblib.load(Config.CALIBRATOR_PATH)

        logger.info('"Models and calibrator loaded successfully."')

    # ── Feature Retrieval ──────────────────────────────────────────────────────
    async def get_features(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Retrieves the latest 15m feature vector from Redis.

        HARD REJECTION POLICY:
          Any missing feature column causes an immediate None return.
          We do NOT fill missing features with defaults — a missing ATR or RSI
          is a data pipeline failure, not a zero-volatility market condition.
        """
        key = f"features:{symbol.lower()}:15m"
        raw = await self._redis.get(key)

        if raw is None:
            logger.warning(f'"Feature key missing in Redis: {key}"')
            return None

        try:
            record = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f'"Feature JSON parse error for {symbol}: {e}"')
            return None

        missing = [f for f in Config.EXPECTED_FEATURES if f not in record]
        if missing:
            logger.error(f'"Hard rejection — missing features for {symbol}: {missing}"')
            return None

        df = pd.DataFrame([record])[Config.EXPECTED_FEATURES]

        # Sanity check: NaN in any cell after strict column selection → reject
        if df.isnull().values.any():
            logger.error(f'"Hard rejection — NaN values in feature vector for {symbol}"')
            return None

        return df

    # ── Prediction + Calibration ───────────────────────────────────────────────
    def predict_calibrated(
        self, df: pd.DataFrame
    ) -> Tuple[int, float, list[float]]:
        """
        Returns (predicted_class, calibrated_confidence, full_proba_vector).

        Architecture:
          1. Single multi-class XGBoost: classes = [0=Short, 1=Neutral, 2=Long]
          2. Raw predict_proba → calibrated via pre-fitted calibrator (Isotonic/Platt)
          3. Predicted class = argmax(calibrated_proba)
          4. Confidence = calibrated_proba[predicted_class]

        Why single model over two binary models:
          - Avoids the mathematical inconsistency of comparing P(long|X) vs P(short|X)
            from independent models with unrelated probability scales.
          - Multi-class softmax output sums to 1.0 — valid probability distribution.
          - Neutral class absorbs low-conviction signals instead of forcing a direction.
        """
        raw_proba = self._model.predict_proba(df)  # shape: (1, 3)

        # Calibrator must be pre-fitted on a held-out validation set
        calibrated_proba = self._calibrator.predict_proba(raw_proba)  # shape: (1, 3)
        proba_vector = calibrated_proba[0].tolist()

        predicted_class = int(np.argmax(proba_vector))
        confidence = round(float(proba_vector[predicted_class]), 4)

        return predicted_class, confidence, [round(p, 4) for p in proba_vector]

    # ── Main Processing Loop ───────────────────────────────────────────────────
    async def _process_request(self, raw_data: str) -> None:
        try:
            req = PriceActionRequest.model_validate_json(raw_data)
        except Exception as e:
            logger.error(f'"Invalid request payload: {e}"')
            return

        features = await self.get_features(req.symbol)
        if features is None:
            # Feature pipeline failure — do not emit a signal
            return

        predicted_class, confidence, proba_vector = self.predict_calibrated(features)

        # Confidence gate: low-conviction predictions are dropped before reaching MasterAI
        if confidence < Config.CONFIDENCE_THRESHOLD:
            logger.info(
                f'"Signal DROPPED — confidence {confidence} below threshold '
                f'{Config.CONFIDENCE_THRESHOLD} for {req.symbol}"'
            )
            return

        signal_out = PriceActionSignal(
            symbol=req.symbol,
            request_id=req.request_id,
            signal=Config.CLASS_SIGNAL_MAP[predicted_class],
            signal_label=Config.CLASS_MAP[predicted_class],
            confidence=confidence,
            raw_proba=proba_vector,
            predicted_class=predicted_class,
            timestamp=time.time(),
        )

        redis_key = f"signal:ml_price_ai:{req.symbol.lower()}"
        await self._redis.setex(
            redis_key,
            Config.RESULT_TTL_SEC,
            signal_out.model_dump_json(),
        )

        logger.info(
            f'"Signal emitted: symbol={req.symbol}, '
            f'label={signal_out.signal_label}, '
            f'confidence={confidence}, '
            f'proba={proba_vector}"'
        )

    async def run(self) -> None:
        self.load_models()
        self._redis = await redis.from_url(Config.REDIS_URL, decode_responses=True)
        logger.info('"MLPriceActionAI listening for requests..."')

        while not self._stop_event.is_set():
            try:
                # Large timeout: KEDA terminates the pod via SIGTERM on scale-down.
                # We do NOT self-terminate on idle — that breaks KEDA's lifecycle contract.
                result = await self._redis.brpop(
                    Config.REQUEST_QUEUE, timeout=Config.BRPOP_TIMEOUT_SEC
                )

                if result is None:
                    # Timeout hit — not an error, just re-enter the loop
                    logger.info('"BRPOP timeout — re-entering listen loop."')
                    continue

                _, raw_data = result
                await self._process_request(raw_data)

            except redis.RedisError as e:
                logger.error(f'"Redis error: {e} — retrying in 2s."')
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f'"Unexpected error: {e}"')
                await asyncio.sleep(1)

        logger.info('"Stop event received. Shutting down gracefully."')
        await self._redis.aclose()

    def shutdown(self, *args) -> None:
        logger.info('"SIGTERM/SIGINT received. Setting stop event."')
        self._stop_event.set()


# ─── Entrypoint ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = MLPriceActionAI()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, agent.shutdown)

    try:
        loop.run_until_complete(agent.run())
    finally:
        loop.close()
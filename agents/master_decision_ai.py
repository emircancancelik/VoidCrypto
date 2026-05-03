from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Deque, Optional
from pydantic import BaseModel, Field, ValidationError

import aio_pika
import numpy as np
import pandas as pd
import redis.asyncio as redis
import xgboost as xgb
from aiohttp import web
from aiormq.exceptions import AMQPConnectionError

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","agent":"MasterDecisionAI","level":"%(levelname)s","msg":%(message)s}',
)
logger = logging.getLogger("MasterDecisionAI")

BASE_DIR = Path(__file__).resolve().parent.parent

def _env(key: str, default: str) -> str:
    return os.getenv(key, default)

class Config:
    REDIS_URL           = _env("REDIS_URL",           "redis://redis-service:6379/0")
    RABBITMQ_URL        = _env("RABBITMQ_URL",        "amqp://guest:guest@rabbitmq-service:5672/")
    CONSUME_QUEUE       = _env("CONSUME_QUEUE",       "market_alerts")
    EXECUTION_QUEUE     = _env("EXECUTION_QUEUE",     "execution_orders")
    DLQ_NAME            = _env("DLQ_NAME",            "market_alerts.dlq")
    AUDITOR_RESULT_TTL  = float(_env("AUDITOR_RESULT_TTL", "0.5"))      # seconds to wait for auditor 
    CONFIDENCE_THRESHOLD= float(_env("CONFIDENCE_THRESHOLD", "0.60"))
    ECE_THRESHOLD       = float(_env("ECE_THRESHOLD",  "0.05"))     # max allowed ECE
    ATR_CONFIDENCE_RATIO= float(_env("ATR_CONFIDENCE_RATIO", "2.0"))# confidence must be ≥ ratio×normalised_ATR
    METRICS_PORT        = int(_env("METRICS_PORT",    "9090"))
    CB_FAILURE_LIMIT    = int(_env("CB_FAILURE_LIMIT","5"))          # circuit breaker trip threshold
    CB_RESET_SECONDS    = int(_env("CB_RESET_SECONDS","30"))
    MODEL_LONG_PATH     = _env("MODEL_LONG_PATH",  str(BASE_DIR / "agents/void_model_15m_long.json"))
    MODEL_SHORT_PATH    = _env("MODEL_SHORT_PATH", str(BASE_DIR / "agents/void_model_15m_short.json"))

EXPECTED_FEATURES = [
    "dist_ema9", "dist_ema21", "dist_ema50", "dist_ema200",
    "macd_hist_pct", "macd_line_pct", "atr_pct",
    "rsi", "obv_roc", "volume_roc",
]
MAX_MISSING_FEATURE_RATIO = 0.3

class CBState(Enum):
    CLOSED   = "CLOSED"    # normal operation
    OPEN     = "OPEN"      # failing – reject fast
    HALF_OPEN= "HALF_OPEN" # probe allowed

class IncomingTrigger(BaseModel):
    symbol: str
    event: str
    details: dict = {}

@dataclass
class CircuitBreaker:
    name: str
    failure_limit: int = Config.CB_FAILURE_LIMIT
    reset_seconds: float = Config.CB_RESET_SECONDS
    _failures: int = field(default=0, init=False)
    _state: CBState = field(default=CBState.CLOSED, init=False)
    _opened_at: float = field(default=0.0, init=False)

    def record_success(self) -> None:
        self._failures = 0
        self._state = CBState.CLOSED

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_limit:
            if self._state != CBState.OPEN:
                logger.error(f'"event":"CB_OPEN","breaker":"{self.name}"')
            self._state = CBState.OPEN
            self._opened_at = time.monotonic()

    def allow_request(self) -> bool:
        if self._state == CBState.CLOSED:
            return True
        if self._state == CBState.OPEN:
            if time.monotonic() - self._opened_at >= self.reset_seconds:
                self._state = CBState.HALF_OPEN
                logger.info(f'"event":"CB_HALF_OPEN","breaker":"{self.name}"')
                return True
            return False
        return True

    @property
    def state(self) -> str:
        return self._state.value

@dataclass
class Metrics:
    messages_consumed: int = 0
    messages_dropped_ece: int = 0
    messages_dropped_contradiction: int = 0
    messages_dropped_confidence: int = 0
    messages_dropped_feature_drift: int = 0
    messages_dispatched: int = 0
    messages_dlq: int = 0
    inference_latency_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    auditor_latency_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def prometheus_text(self) -> str:
        lat_p99 = float(np.percentile(self.inference_latency_ms, 99)) if self.inference_latency_ms else 0.0
        aud_p99 = float(np.percentile(self.auditor_latency_ms, 99))   if self.auditor_latency_ms  else 0.0
        lines = [
            "# HELP voidcrypto_master_messages_consumed_total Total messages consumed",
            "# TYPE voidcrypto_master_messages_consumed_total counter",
            f"voidcrypto_master_messages_consumed_total {self.messages_consumed}",
            f"voidcrypto_master_messages_dropped_ece_total {self.messages_dropped_ece}",
            f"voidcrypto_master_messages_dropped_contradiction_total {self.messages_dropped_contradiction}",
            f"voidcrypto_master_messages_dropped_confidence_total {self.messages_dropped_confidence}",
            f"voidcrypto_master_messages_dropped_feature_drift_total {self.messages_dropped_feature_drift}",
            f"voidcrypto_master_messages_dispatched_total {self.messages_dispatched}",
            f"voidcrypto_master_messages_dlq_total {self.messages_dlq}",
            f"voidcrypto_master_inference_p99_ms {lat_p99:.2f}",
            f"voidcrypto_master_auditor_p99_ms {aud_p99:.2f}",
        ]
        return "\n".join(lines) + "\n"

class AuditorGate:
    """
    Inline Auditor layer between Master Decision and Risk AI.

    Responsibilities (mirrors the Auditor AI spec):
      1. ECE check  – calibration error of the winning model must be ≤ ECE_THRESHOLD
      2. ATR cross-check – confidence must be proportional to current volatility
      3. Contradiction detection – if sub-agent signals cached in Redis contradict
         the master direction, hard DROP.

    If AuditorAI pod is deployed (publishes to Redis key auditor_result:<correlation_id>),
    we wait up to AUDITOR_RESULT_TTL seconds for its verdict and honour it.
    Otherwise we fall back to local heuristics so the pipeline is not blocked.
    """

    def __init__(self, redis_client: redis.Redis, metrics: Metrics):
        self._redis = redis_client
        self._metrics = metrics

    async def validate(
        self,
        correlation_id: str,
        symbol: str,
        direction: str,
        confidence: float,
        features: pd.DataFrame,
        calibration_ece: float,
    ) -> tuple[bool, str]:
        # 1. ECE gate
        if calibration_ece > Config.ECE_THRESHOLD:
            self._metrics.messages_dropped_ece += 1
            return False, f"ECE={calibration_ece:.4f} exceeds threshold={Config.ECE_THRESHOLD}"

        # 2. ATR volatility cross-check
        atr_pct = float(features["atr_pct"].iloc[0])
        # Normalised ATR: if market is wild, require higher confidence
        required_confidence = min(0.95, Config.CONFIDENCE_THRESHOLD + atr_pct * Config.ATR_CONFIDENCE_RATIO)
        if confidence < required_confidence:
            self._metrics.messages_dropped_ece += 1
            return False, (
                f"ATR-adjusted confidence gate failed: "
                f"conf={confidence:.4f} < required={required_confidence:.4f} (atr_pct={atr_pct:.4f})"
            )

        # 3. Sub-agent contradiction check (Whale vs TA signals in Redis)
        contradiction, reason = await self._check_subagent_contradiction(symbol, direction)
        if contradiction:
            self._metrics.messages_dropped_contradiction += 1
            return False, reason

        # 4. Optional: wait for external AuditorAI pod verdict
        external_verdict = await self._poll_external_auditor(correlation_id)
        if external_verdict is not None:
            if not external_verdict.get("approved", False):
                reason = external_verdict.get("reason", "AUDITOR_REJECTED")
                self._metrics.messages_dropped_ece += 1
                return False, f"ExternalAuditor: {reason}"

        return True, "OK"

    # ── private ───────────────────────────────────────────────────
    async def _check_subagent_contradiction(
        self, symbol: str, direction: str
    ) -> tuple[bool, str]:
        keys_to_check = [
            f"agent_signal:ta_ai:{symbol.lower()}",
            f"agent_signal:whale_ai:{symbol.lower()}",
        ]
        contradictions: list[str] = []
        for key in keys_to_check:
            try:
                raw = await self._redis.get(key)
                if raw is None:
                    continue 
                signal = json.loads(raw)
                agent_direction = signal.get("direction", "").upper()
                agent_confidence = float(signal.get("confidence", 0.0))
                # Only flag if the sub-agent is confident AND points the other way
                if agent_direction and agent_direction != direction and agent_confidence >= Config.CONFIDENCE_THRESHOLD:
                    contradictions.append(f"{key}→{agent_direction}({agent_confidence:.2f})")
            except Exception as exc:
                logger.warning(f'"event":"SUBAGENT_CHECK_ERROR","key":"{key}","error":"{exc}"')

        if contradictions:
            return True, f"SubAgent contradiction: master={direction} vs [{', '.join(contradictions)}]"
        return False, ""

    async def _poll_external_auditor(self, correlation_id: str) -> Optional[dict]:
        key = f"auditor_result:{correlation_id}"
        deadline = time.monotonic() + Config.AUDITOR_RESULT_TTL
        while time.monotonic() < deadline:
            raw = await self._redis.get(key)
            if raw:
                await self._redis.delete(key)  # consume once
                return json.loads(raw)
            await asyncio.sleep(0.1)
        return None 


# ──────────────────────────── FEATURE VALIDATION ─────────────────
def validate_and_build_features(raw_data: dict) -> tuple[pd.DataFrame, float]:
    df = pd.DataFrame([raw_data])
    imputed = 0
    for col in EXPECTED_FEATURES:
        if col not in df.columns or pd.isna(df[col].iloc[0]):
            df[col] = 0.0
            imputed += 1

    missing_ratio = imputed / len(EXPECTED_FEATURES)
    if missing_ratio > MAX_MISSING_FEATURE_RATIO:
        raise ValueError(
            f"Feature drift: {imputed}/{len(EXPECTED_FEATURES)} features missing "
            f"(ratio={missing_ratio:.2f} > threshold={MAX_MISSING_FEATURE_RATIO})"
        )

    ece_proxy = missing_ratio  # 0.0 = perfect, 1.0 = all missing
    return df[EXPECTED_FEATURES], ece_proxy

class MasterOrchestrator:
    def __init__(self):
        self.model_long:  Optional[xgb.XGBClassifier] = None
        self.model_short: Optional[xgb.XGBClassifier] = None
        self.redis_client:        Optional[redis.Redis]            = None
        self.rabbitmq_conn:       Optional[aio_pika.RobustConnection] = None
        self.rabbitmq_channel:    Optional[aio_pika.Channel]       = None
        self.execution_exchange:  Optional[aio_pika.Exchange]      = None
        self.dlq_exchange:        Optional[aio_pika.Exchange]      = None
        self.auditor:             Optional[AuditorGate]            = None
        self.metrics = Metrics()
        self._cb_redis   = CircuitBreaker("redis")
        self._cb_rabbitmq= CircuitBreaker("rabbitmq")
        self.is_running  = True
        self._metrics_app: Optional[web.Application] = None
        self._metrics_runner: Optional[web.AppRunner] = None

    def load_models(self) -> None:
        logger.info('"event":"MODEL_LOAD_START"')
        try:
            self.model_long  = xgb.XGBClassifier()
            self.model_long.load_model(Config.MODEL_LONG_PATH)
            self.model_short = xgb.XGBClassifier()
            self.model_short.load_model(Config.MODEL_SHORT_PATH)
            logger.info('"event":"MODEL_LOAD_COMPLETE"')
        except Exception as exc:
            logger.error(f'"event":"MODEL_LOAD_FAILED","error":"{exc}"')
            raise SystemExit(1) from exc

    async def connect_infrastructure(self) -> None:
        # Redis
        self.redis_client = redis.from_url(Config.REDIS_URL, decode_responses=True)
        self.auditor = AuditorGate(self.redis_client, self.metrics)

        while self.is_running:
            try:
                logger.info('"event":"INFRA_CONNECT_ATTEMPT","target":"rabbitmq"')
                self.rabbitmq_conn    = await aio_pika.connect_robust(Config.RABBITMQ_URL, timeout=10)
                self.rabbitmq_channel = await self.rabbitmq_conn.channel()
                await self.rabbitmq_channel.set_qos(prefetch_count=10)

                # Declare DLQ first
                dlq = await self.rabbitmq_channel.declare_queue(Config.DLQ_NAME, durable=True)

                # Declare main queue with x-dead-letter routing to DLQ
                await self.rabbitmq_channel.declare_queue(
                    Config.EXECUTION_QUEUE,
                    durable=True,
                    arguments={
                        "x-dead-letter-exchange": "",
                        "x-dead-letter-routing-key": Config.DLQ_NAME,
                    },
                )

                self.execution_exchange = self.rabbitmq_channel.default_exchange
                self.dlq_exchange       = self.rabbitmq_channel.default_exchange
                self._cb_rabbitmq.record_success()
                logger.info('"event":"INFRA_CONNECT_OK"')
                break
            except (AMQPConnectionError, OSError) as exc:
                self._cb_rabbitmq.record_failure()
                logger.warning(f'"event":"INFRA_CONNECT_RETRY","reason":"RabbitMQ not ready","error":"{exc}"')
                await asyncio.sleep(3)

    async def start_metrics_server(self) -> None:
        async def handle_metrics(request: web.Request) -> web.Response:
            return web.Response(text=self.metrics.prometheus_text(), content_type="text/plain")

        async def handle_health(request: web.Request) -> web.Response:
            return web.Response(text='{"status":"ok"}', content_type="application/json")

        self._metrics_app = web.Application()
        self._metrics_app.router.add_get("/metrics", handle_metrics)
        self._metrics_app.router.add_get("/healthz",  handle_health)
        self._metrics_runner = web.AppRunner(self._metrics_app)
        await self._metrics_runner.setup()
        site = web.TCPSite(self._metrics_runner, "0.0.0.0", Config.METRICS_PORT)
        await site.start()
        logger.info(f'"event":"METRICS_SERVER_STARTED","port":{Config.METRICS_PORT}')

    async def _get_features(self, symbol: str) -> tuple[pd.DataFrame, float]:
        if not self._cb_redis.allow_request():
            raise RuntimeError("CircuitBreaker[redis] OPEN – skipping Redis call")
        try:
            raw_data = await self.redis_client.get(f"model_features:{symbol.lower()}:15m")
            self._cb_redis.record_success()
        except Exception as exc:
            self._cb_redis.record_failure()
            raise RuntimeError(f"Redis GET failed: {exc}") from exc

        if not raw_data:
            raise ValueError(f"Missing feature matrix for {symbol}")

        data = json.loads(raw_data)
        return validate_and_build_features(data)  
    
    async def _infer(self, features_df: pd.DataFrame) -> tuple[float, float]:
        t0 = time.monotonic()
        prob_long, prob_short = await asyncio.to_thread(
            self._run_inference_sync, features_df
        )
        latency_ms = (time.monotonic() - t0) * 1000
        self.metrics.inference_latency_ms.append(latency_ms)
        return prob_long, prob_short

    def _run_inference_sync(self, features_df: pd.DataFrame) -> tuple[float, float]:
        prob_long  = float(self.model_long.predict_proba(features_df)[0][1])
        prob_short = float(self.model_short.predict_proba(features_df)[0][1])
        return prob_long, prob_short

    # ── dispatch to Risk AI ───────────────────────────────────────
    async def _dispatch_to_risk_ai(
        self,
        symbol: str,
        direction: str,
        price: float,
        confidence: float,
        trigger_event: str,
        correlation_id: str,
    ) -> None:
        if not self._cb_rabbitmq.allow_request():
            raise RuntimeError("CircuitBreaker[rabbitmq] OPEN – cannot dispatch")

        payload = {
            "correlation_id": correlation_id,
            "symbol":         symbol,
            "direction":      direction,
            "trigger_price":  price,
            "confidence":     round(confidence, 4),
            "trigger_source": trigger_event,
            "action":         "EVALUATE_ENTRY",
        }
        try:
            await self.execution_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(payload).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    message_id=correlation_id,
                ),
                routing_key=Config.EXECUTION_QUEUE,
            )
            self._cb_rabbitmq.record_success()
            self.metrics.messages_dispatched += 1
            logger.warning(f'"event":"ORDER_DISPATCHED","payload":{json.dumps(payload)}')
        except Exception as exc:
            self._cb_rabbitmq.record_failure()
            raise RuntimeError(f"RabbitMQ publish failed: {exc}") from exc

    async def _send_to_dlq(self, raw_body: bytes, reason: str) -> None:
        self.metrics.messages_dlq += 1
        try:
            await self.dlq_exchange.publish(
                aio_pika.Message(
                    body=raw_body,
                    headers={"x-drop-reason": reason[:255]},
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=Config.DLQ_NAME,
            )
            logger.error(f'"event":"MESSAGE_DLQ","reason":"{reason}"')
        except Exception as exc:
            logger.error(f'"event":"DLQ_PUBLISH_FAILED","error":"{exc}"')

    async def execute_consensus(self, message: aio_pika.IncomingMessage) -> None:
        self.metrics.messages_consumed += 1
        raw_body = message.body
        correlation_id = message.message_id or f"cid_{time.time_ns()}"

        try:
            payload = json.loads(raw_body.decode()) 
            trigger = IncomingTrigger(**payload) 
            symbol = trigger.symbol
            trigger_event = trigger.event
            details = trigger.details
            price = float(details.get("mid_price", 0.0))
        except (json.JSONDecodeError, ValidationError, KeyError) as exc:
            await self._send_to_dlq(raw_body, f"MALFORMED_PAYLOAD: {exc}")
            return

        # 1. Feature retrieval + drift check
        try:
            features_df, ece_proxy = await self._get_features(symbol)
        except ValueError as exc:
            self.metrics.messages_dropped_feature_drift += 1
            await self._send_to_dlq(raw_body, f"FEATURE_DRIFT: {exc}")
            logger.error(f'"event":"FEATURE_DRIFT","symbol":"{symbol}","error":"{exc}"')
            return
        except RuntimeError as exc:
            # Circuit breaker open – requeue for later
            await self._send_to_dlq(raw_body, f"INFRA_UNAVAILABLE: {exc}")
            return

        # 2. Inference
        try:
            prob_long, prob_short = await self._infer(features_df)
        except Exception as exc:
            await self._send_to_dlq(raw_body, f"INFERENCE_ERROR: {exc}")
            logger.error(f'"event":"INFERENCE_ERROR","symbol":"{symbol}","error":"{exc}"')
            return

        logger.info(
            f'"event":"INFERENCE_RESULT","symbol":"{symbol}",'
            f'"prob_long":{prob_long:.4f},"prob_short":{prob_short:.4f},'
            f'"ece_proxy":{ece_proxy:.4f}'
        )

        # 3. Confidence gate (fast path before Auditor)
        if prob_long < Config.CONFIDENCE_THRESHOLD and prob_short < Config.CONFIDENCE_THRESHOLD:
            self.metrics.messages_dropped_confidence += 1
            logger.info(
                f'"event":"SIGNAL_BELOW_THRESHOLD","symbol":"{symbol}",'
                f'"prob_long":{prob_long:.4f},"prob_short":{prob_short:.4f}'
            )
            return  # ack silently – no trade signal

        # 4. Determine direction
        if prob_long >= Config.CONFIDENCE_THRESHOLD and prob_long > prob_short:
            direction  = "LONG"
            confidence = prob_long
        else:
            direction  = "SHORT"
            confidence = prob_short

        # 5. Auditor gate  ← This is the core architectural requirement
        t0 = time.monotonic()
        approved, reason = await self.auditor.validate(
            correlation_id=correlation_id,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            features=features_df,
            calibration_ece=ece_proxy,
        )
        self.metrics.auditor_latency_ms.append((time.monotonic() - t0) * 1000)

        if not approved:
            logger.error(
                f'"event":"AUDITOR_DROP","symbol":"{symbol}",'
                f'"direction":"{direction}","reason":"{reason}"'
            )
            # Publish drop event so other agents can observe (optional downstream)
            await self._publish_audit_drop_event(symbol, direction, confidence, reason)
            return

        # 6. Dispatch to Risk AI
        try:
            await self._dispatch_to_risk_ai(
                symbol=symbol,
                direction=direction,
                price=price,
                confidence=confidence,
                trigger_event=trigger_event,
                correlation_id=correlation_id,
            )
        except RuntimeError as exc:
            await self._send_to_dlq(raw_body, f"DISPATCH_FAILED: {exc}")

    async def _publish_audit_drop_event(
        self, symbol: str, direction: str, confidence: float, reason: str
    ) -> None:
        event = {
            "event":      "AUDITOR_DROP",
            "symbol":     symbol,
            "direction":  direction,
            "confidence": round(confidence, 4),
            "reason":     reason,
            "ts":         time.time(),
        }
        try:
            await self.redis_client.setex(
                f"audit_drop:{symbol.lower()}",
                60,
                json.dumps(event),
            )
        except Exception as exc:
            logger.warning(f'"event":"AUDIT_DROP_PUBLISH_FAILED","error":"{exc}"')

    # ── graceful shutdown ─────────────────────────────────────────
    async def shutdown(self) -> None:
        logger.info('"event":"SHUTDOWN_START"')
        self.is_running = False
        if self._metrics_runner:
            await self._metrics_runner.cleanup()
        if self.rabbitmq_conn:
            await self.rabbitmq_conn.close()
        if self.redis_client:
            await self.redis_client.aclose()
        logger.info('"event":"SHUTDOWN_COMPLETE"')


# ──────────────────────────── ENTRYPOINT ─────────────────────────
async def main() -> None:
    orchestrator = MasterOrchestrator()
    orchestrator.load_models()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    try:
        await orchestrator.connect_infrastructure()
        await orchestrator.start_metrics_server()

        consume_queue = await orchestrator.rabbitmq_channel.declare_queue(
            Config.CONSUME_QUEUE, durable=True
        )

        async def on_message(message: aio_pika.IncomingMessage) -> None:
            async with message.process(requeue=False):  # nack → DLQ via x-dead-letter
                await orchestrator.execute_consensus(message)

        await consume_queue.consume(on_message)
        logger.info(
            f'"event":"CONSUMER_READY","queue":"{Config.CONSUME_QUEUE}",'
            f'"metrics_port":{Config.METRICS_PORT}'
        )
        await stop_event.wait()

    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
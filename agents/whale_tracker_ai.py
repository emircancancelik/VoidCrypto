import asyncio
import json
import logging
import os
import signal
import sys
import time
import aiohttp
import redis.asyncio as aioredis

from contextlib import asynccontextmanager
from typing import Optional
from aio_pika import connect_robust, ExchangeType, Message
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

# ─── Structured JSON logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","agent":"WhaleTrackerAI","lvl":"%(levelname)s","body":%(message)s}',
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("WhaleTrackerAI")


# ─── Configuration ─────────────────────────────────────────────────────────────
class WhaleConfig(BaseSettings):
    # Infrastructure
    REDIS_URL: str = "redis://redis-service:6379/0"
    RABBITMQ_URL: str = "amqp://guest:guest@rabbitmq-service:5672/"
    RABBITMQ_EXCHANGE: str = "voidcrypto.signals"
    RABBITMQ_QUEUE: str = "market_alerts"
    # Whale Alert API
    WHALE_API_KEY: str = Field(..., description="Whale Alert API key — required")
    WHALE_API_BASE: str = "https://api.whale-alert.io/v1"
    # How far back to look on each poll (seconds). Should match KEDA cron interval.
    WHALE_LOOKBACK_SECONDS: int = 300  # 5 min

    # Signal thresholds
    WHALE_THRESHOLD_USD: float = 500_000.0
    # Confidence normalisation ceiling: transfers at this USD value → confidence=1.0
    WHALE_CONFIDENCE_CEILING_USD: float = 10_000_000.0

    # Symbols to watch (must match Whale Alert blockchain symbol names)
    WATCHED_SYMBOLS: list[str] = ["BTC", "ETH", "SOL","ACH","GALA","GRT"]

    # Redis TTL for stored signal (seconds). Must exceed Master AI poll interval.
    SIGNAL_TTL_SECONDS: int = 600

    # Known exchange wallet labels (lowercase). Extend as needed.
    EXCHANGE_LABELS: list[str] = [
        "binance", "coinbase", "kraken", "okx", "bybit", "bitfinex",
        "huobi", "kucoin", "gemini", "ftx", "bitmex", "deribit",
        "exchange", "hot wallet", "trading",
    ]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# ─── Data Models ───────────────────────────────────────────────────────────────
class WhaleSignal(BaseModel):
    agent: str = "whale_ai"
    symbol: str
    direction: str           # "LONG" | "SHORT"
    classification: int      # 1 = bullish pressure, 0 = bearish pressure
    confidence: float        # [0.0, 1.0] — calibrated, NOT raw
    amount_usd: float
    tx_hash: Optional[str] = None
    from_label: Optional[str] = None
    to_label: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return round(min(1.0, max(0.0, v)), 4)

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        if v not in ("LONG", "SHORT"):
            raise ValueError(f"Invalid direction: {v}. Must be LONG or SHORT.")
        return v


class WhaleRawTransaction(BaseModel):
    """Normalised schema for a single Whale Alert transaction."""
    id: str
    blockchain: str
    symbol: str
    amount_usd: float
    from_owner: str = ""
    from_owner_type: str = ""
    to_owner: str = ""
    to_owner_type: str = ""
    hash: Optional[str] = None
    timestamp: int


# ─── Core Agent ────────────────────────────────────────────────────────────────
class WhaleTrackerAI:
    def __init__(self, config: WhaleConfig):
        self.cfg = config
        self.redis: Optional[aioredis.Redis] = None
        self.rmq_connection = None
        self.rmq_channel = None
        self.rmq_exchange = None
        self._seen_tx_ids: set[str] = set()  # dedup within single run

    # ── Infrastructure ────────────────────────────────────────────────────────
    async def _connect_redis(self) -> None:
        self.redis = await aioredis.from_url(
            self.cfg.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )
        await self.redis.ping()
        logger.info('"event":"REDIS_CONNECTED"')

    async def _connect_rabbitmq(self) -> None:
        self.rmq_connection = await connect_robust(
            self.cfg.RABBITMQ_URL,
            timeout=10,
        )
        self.rmq_channel = await self.rmq_connection.channel()
        self.rmq_exchange = await self.rmq_channel.declare_exchange(
            self.cfg.RABBITMQ_EXCHANGE,
            ExchangeType.TOPIC,
            durable=True,
        )
        logger.info('"event":"RABBITMQ_CONNECTED"')

    async def _disconnect(self) -> None:
        if self.redis:
            await self.redis.aclose()
        if self.rmq_connection and not self.rmq_connection.is_closed:
            await self.rmq_connection.close()
        logger.info('"event":"CONNECTIONS_CLOSED"')

    # ── Whale Alert API ───────────────────────────────────────────────────────
    async def _fetch_transactions(
        self, session: aiohttp.ClientSession
    ) -> list[WhaleRawTransaction]:
        """
        Fetches large transactions from Whale Alert API.
        Returns a deduplicated list of WhaleRawTransaction objects.
        """
        since = int(time.time()) - self.cfg.WHALE_LOOKBACK_SECONDS
        params = {
            "api_key": self.cfg.WHALE_API_KEY,
            "min_value": int(self.cfg.WHALE_THRESHOLD_USD),
            "start": since,
            "currency": ",".join(c.lower() for c in self.cfg.WATCHED_SYMBOLS),
        }
        url = f"{self.cfg.WHALE_API_BASE}/transactions"

        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 429:
                    logger.warning('"event":"RATE_LIMITED","action":"skipping_poll"')
                    return []
                resp.raise_for_status()
                payload = await resp.json()
        except aiohttp.ClientResponseError as e:
            logger.error(f'"event":"API_HTTP_ERROR","status":{e.status},"msg":"{e.message}"')
            return []
        except asyncio.TimeoutError:
            logger.error('"event":"API_TIMEOUT"')
            return []

        raw_txs = payload.get("transactions", [])
        result = []
        for tx in raw_txs:
            tx_id = str(tx.get("id", ""))
            if tx_id in self._seen_tx_ids:
                continue
            self._seen_tx_ids.add(tx_id)
            try:
                result.append(WhaleRawTransaction(
                    id=tx_id,
                    blockchain=tx.get("blockchain", ""),
                    symbol=tx.get("symbol", "").upper(),
                    amount_usd=float(tx.get("amount_usd", 0)),
                    from_owner=str(tx.get("from", {}).get("owner", "") or "").lower(),
                    from_owner_type=str(tx.get("from", {}).get("owner_type", "") or "").lower(),
                    to_owner=str(tx.get("to", {}).get("owner", "") or "").lower(),
                    to_owner_type=str(tx.get("to", {}).get("owner_type", "") or "").lower(),
                    hash=tx.get("hash"),
                    timestamp=int(tx.get("timestamp", time.time())),
                ))
            except Exception as e:
                logger.warning(f'"event":"TX_PARSE_ERROR","tx_id":"{tx_id}","err":"{e}"')
        return result

    # ── Classification ────────────────────────────────────────────────────────
    def _is_exchange_label(self, label: str) -> bool:
        """
        Checks against known exchange labels.
        Robust against partial matches (e.g. 'binance hot wallet 3').
        """
        label_lower = label.lower()
        return any(ex_label in label_lower for ex_label in self.cfg.EXCHANGE_LABELS)

    def _classify_direction(self, tx: WhaleRawTransaction) -> Optional[str]:
        """
        Returns 'LONG', 'SHORT', or None (discard).

        Rules:
        - Wallet → Exchange  = sell pressure → SHORT (classification=0)
        - Exchange → Wallet  = accumulation → LONG  (classification=1)
        - Exchange → Exchange= internal move → None (noise, discard)
        - Wallet → Wallet    = OTC or cold storage → None (ambiguous, discard)

        Priority: owner_type field first (authoritative), then owner name label.
        """
        from_is_exchange = (
            tx.from_owner_type == "exchange"
            or self._is_exchange_label(tx.from_owner)
        )
        to_is_exchange = (
            tx.to_owner_type == "exchange"
            or self._is_exchange_label(tx.to_owner)
        )

        if to_is_exchange and not from_is_exchange:
            return "SHORT"
        if from_is_exchange and not to_is_exchange:
            return "LONG"
        return None  # exchange-to-exchange or wallet-to-wallet

    def _calibrate_confidence(self, amount_usd: float) -> float:
        """
        Monotonically increasing, logarithmic confidence score.
        Avoids the original linear formula which penalises moderately large
        transactions and gives equal weight to very different order-of-magnitudes.

        Formula: log(amount / threshold) / log(ceiling / threshold)
        Result clipped to [0.05, 1.0] — never returns zero for a passing tx.
        """
        import math
        threshold = self.cfg.WHALE_THRESHOLD_USD
        ceiling = self.cfg.WHALE_CONFIDENCE_CEILING_USD
        if amount_usd <= threshold:
            return 0.05
        log_score = math.log(amount_usd / threshold) / math.log(ceiling / threshold)
        return round(min(1.0, max(0.05, log_score)), 4)

    def _build_signal(self, tx: WhaleRawTransaction, direction: str) -> WhaleSignal:
        classification = 1 if direction == "LONG" else 0
        return WhaleSignal(
            symbol=tx.symbol,
            direction=direction,
            classification=classification,
            confidence=self._calibrate_confidence(tx.amount_usd),
            amount_usd=tx.amount_usd,
            tx_hash=tx.hash,
            from_label=tx.from_owner or None,
            to_label=tx.to_owner or None,
            timestamp=float(tx.timestamp),
        )

    # ── Output: Redis + RabbitMQ ──────────────────────────────────────────────
    async def _publish_signal(self, signal: WhaleSignal) -> None:
        signal_json = signal.model_dump_json()
        symbol_lower = signal.symbol.lower()

        # 1. Redis State Yazımı (Master AI / Auditor kontrolü için)
        redis_key = f"agent_signal:whale_ai:{symbol_lower}"
        await self.redis.setex(redis_key, self.cfg.SIGNAL_TTL_SECONDS, signal_json)

        # 2. HATA 2 DÜZELTMESİ: Mesajı Master AI kuyruğuna gönder
        # Default exchange kullanarak doğrudan kuyruk adını routing_key olarak veriyoruz.
        await self.rmq_channel.default_exchange.publish(
            Message(
                body=signal_json.encode(),
                content_type="application/json",
                delivery_mode=2,  # Persistent
            ),
            routing_key=self.cfg.RABBITMQ_QUEUE,
        )

        logger.warning(json.dumps({
            "event": "WHALE_SIGNAL_PUBLISHED",
            "symbol": signal.symbol,
            "direction": signal.direction,
            "classification": signal.classification,
            "confidence": signal.confidence,
            "amount_usd": round(signal.amount_usd, 0),
            "from": signal.from_label,
            "to": signal.to_label,
            "tx_hash": signal.tx_hash
        }))

    # ── Main execution ────────────────────────────────────────────────────────
    async def run(self) -> None:
        """
        Single-shot execution model.
        Agent wakes, connects, polls, publishes all signals, disconnects, exits.
        KEDA is responsible for scheduling re-invocation.
        """
        try:
            await self._connect_redis()
            await self._connect_rabbitmq()

            async with aiohttp.ClientSession(
                headers={"User-Agent": "VoidCrypto-WhaleTrackerAI/2.0"}
            ) as session:
                transactions = await self._fetch_transactions(session)

            if not transactions:
                logger.info('"event":"NO_WHALE_TXS","action":"exiting_cleanly"')
                return

            processed = 0
            discarded = 0
            for tx in transactions:
                if tx.symbol not in self.cfg.WATCHED_SYMBOLS:
                    continue
                dedup_key = f"processed_tx:{tx.id}"
                acquired = await self.redis.set(dedup_key, "1", ex=600, nx=True)
                
                if not acquired:
                    skipped_duplicate += 1
                    continue

                direction = self._classify_direction(tx)
                if direction is None:
                    continue

                signal = self._build_signal(tx, direction)
                await self._publish_signal(signal)
                processed += 1

            logger.info(
                f'"event":"RUN_COMPLETE","processed":{processed},'
                f'"skipped_duplicate":{skipped_duplicate}'
            )

        except Exception as e:
            logger.exception(f'"event":"FATAL_ERROR","err":"{e}"')
            sys.exit(1)
        finally:
            await self._disconnect()


# ─── Entry point ───────────────────────────────────────────────────────────────
def _handle_sigterm(signum, frame):
    logger.info('"event":"SIGTERM_RECEIVED","action":"exiting"')
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_sigterm)
    cfg = WhaleConfig()
    agent = WhaleTrackerAI(cfg)
    asyncio.run(agent.run())
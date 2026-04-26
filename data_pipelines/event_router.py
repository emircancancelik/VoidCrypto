"""
VoidCrypto - SmartEventRouter
Azure Container Apps / KEDA maliyet minimizasyon katmanı.

Tasarım ilkeleri:
- asyncio.Lock ile race-condition elimine edildi
- KEDA uyanış threshold'u: min_signals_to_wake ile kontrol edilir
- Adaptive batching: volatilite yüksekse window küçülür, sakinse büyür
- RabbitMQ circuit-breaker: bağlantı yoksa Redis pub/sub fallback devreye girer
- Memory baskısı: prices listesi yerine online Welford mean/variance tutulur
- Azure cost tag: her payload'a cost_tier eklenir (KEDA filter için)
"""

import asyncio
import json
import logging
import time
import math
from dataclasses import dataclass, field
from typing import Optional

import aio_pika
import redis.asyncio as redis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [EventRouter] %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Welford Online İstatistik — prices listesi tutmadan mean/variance hesapla
# ---------------------------------------------------------------------------
@dataclass
class WelfordAccumulator:
    """O(1) bellek ile online mean ve variance (Welford algoritması)."""
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0       # variance için running sum
    max_change: float = 0.0
    min_price: float = float("inf")
    max_price: float = float("-inf")
    first_seen: float = field(default_factory=time.time)

    def update(self, price: float, change_pct: float) -> None:
        self.count += 1
        delta = price - self.mean
        self.mean += delta / self.count
        delta2 = price - self.mean
        self.M2 += delta * delta2

        if abs(change_pct) > abs(self.max_change):
            self.max_change = change_pct
        if price < self.min_price:
            self.min_price = price
        if price > self.max_price:
            self.max_price = price

    @property
    def variance(self) -> float:
        return self.M2 / self.count if self.count > 1 else 0.0

    @property
    def stddev(self) -> float:
        return math.sqrt(self.M2 / self.count) if self.count > 1 else 0.0


# ---------------------------------------------------------------------------
# KEDA maliyet tier: hangi seviyede Azure container uyanacak?
# ---------------------------------------------------------------------------
def classify_cost_tier(signal_count: int, max_change_pct: float) -> str:
    """
    Azure Container Apps'te KEDA bu alanı filtreler.
    Düşük tier = container uyanmaz (scale-to-zero korunur).
    """
    abs_change = abs(max_change_pct)
    if abs_change >= 3.0 or signal_count >= 500:
        return "CRITICAL"    # TA + DL + Master + Auditor uyan
    elif abs_change >= 1.5 or signal_count >= 200:
        return "HIGH"        # TA + DL uyan
    elif abs_change >= 0.5 or signal_count >= 50:
        return "MEDIUM"      # Sadece TA uyan
    else:
        return "LOW"         # HİÇ container uyanmaz → Azure maliyeti sıfır


# ---------------------------------------------------------------------------
# Ana Router
# ---------------------------------------------------------------------------
class SmartEventRouter:
    """
    Yüksek frekanslı piyasa sinyallerini toplar, analiz eder,
    sadece eşik geçilince KEDA üzerinden Azure container'ları uyandırır.

    Parametreler
    ------------
    base_batch_window_ms   : Normal piyasada bekleme penceresi (ms)
    min_batch_window_ms    : Yüksek volatilitede minimum pencere (ms)
    volatility_scale_factor: max_change > bu değeri geçerse window yarıya iner
    min_signals_to_wake    : Kaç sinyal birikmeden RabbitMQ'ya GİTMEZ
    min_change_pct_to_wake : Bu yüzde altında Azure uyanmaz (maliyet sıfır)
    rabbitmq_routing_key   : KEDA'nın dinlediği queue adı
    redis_fallback_channel : RabbitMQ down ise Redis pub/sub channel
    circuit_breaker_limit  : Art arda kaç RabbitMQ hatasında devre kesilsin
    """

    def __init__(
        self,
        base_batch_window_ms: int = 5000,
        min_batch_window_ms: int = 1000,
        volatility_scale_factor: float = 2.0,
        min_signals_to_wake: int = 10,
        min_change_pct_to_wake: float = 0.3,
        rabbitmq_routing_key: str = "wakeup_queue",
        redis_fallback_channel: str = "vc:wakeup:fallback",
        circuit_breaker_limit: int = 5,
    ):
        self.base_window = base_batch_window_ms / 1000.0
        self.min_window = min_batch_window_ms / 1000.0
        self.volatility_scale_factor = volatility_scale_factor
        self.min_signals_to_wake = min_signals_to_wake
        self.min_change_pct_to_wake = min_change_pct_to_wake
        self.routing_key = rabbitmq_routing_key
        self.redis_fallback_channel = redis_fallback_channel
        self.circuit_breaker_limit = circuit_breaker_limit

        # Bağlantılar — dışarıdan inject edilir
        self.redis: Optional[redis.Redis] = None
        self.rabbitmq_channel: Optional[aio_pika.abc.AbstractChannel] = None

        # Buffer ve senkronizasyon
        self._buffer: dict[str, WelfordAccumulator] = {}
        self._lock = asyncio.Lock()                  # race-condition kalkanı
        self._flush_task: Optional[asyncio.Task] = None

        # Circuit-breaker state
        self._rabbitmq_errors: int = 0
        self._circuit_open: bool = False             # True = RabbitMQ bypass

        # Prometheus/metrics stub (gerçek ortamda prometheus_client bağlanır)
        self._metrics = {
            "signals_received": 0,
            "flushes_triggered": 0,
            "keda_wakeups": 0,
            "keda_suppressed": 0,
            "circuit_breaker_trips": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def register_signal(self, symbol: str, price: float, change_pct: float) -> None:
        """
        Saniyede binlerce kez çağrılabilir. Non-blocking, lock minimal.
        """
        async with self._lock:
            if symbol not in self._buffer:
                self._buffer[symbol] = WelfordAccumulator(first_seen=time.time())
            self._buffer[symbol].update(price, change_pct)
            self._metrics["signals_received"] += 1

            # Flush motoru yoksa başlat
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(
                    self._flush_loop(),
                    name=f"flush_loop_{int(time.time())}"
                )

    async def close(self) -> None:
        """Graceful shutdown — bekleyen flush'u tamamla."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Kalan buffer'ı zorla flush et
        await self._execute_flush(force=True)
        logger.info("EventRouter kapatıldı. Metrikler: %s", self._metrics)

    # ------------------------------------------------------------------
    # Adaptive Batch Window
    # ------------------------------------------------------------------

    def _compute_adaptive_window(self) -> float:
        """
        Tüm sembollerdeki max change'e göre pencere süresini daralt.
        Piyasa sakinse: base_window (5s) → az uyanış → düşük maliyet.
        Piyasa volatilifse: min_window (1s) → hızlı tepki.
        """
        if not self._buffer:
            return self.base_window
        max_global_change = max(abs(acc.max_change) for acc in self._buffer.values())
        if max_global_change >= self.volatility_scale_factor:
            # Volatilite eşiği geçildi: pencereyi ölçekle (lineer interpolasyon)
            ratio = min(max_global_change / (self.volatility_scale_factor * 3), 1.0)
            window = self.base_window - ratio * (self.base_window - self.min_window)
            return max(window, self.min_window)
        return self.base_window

    # ------------------------------------------------------------------
    # Flush Döngüsü
    # ------------------------------------------------------------------

    async def _flush_loop(self) -> None:
        """
        Adaptive window boyunca bekle, ardından flush et.
        Buffer boşalana kadar döngü sürer.
        """
        while True:
            window = self._compute_adaptive_window()
            logger.debug("Adaptive batch window: %.2fs", window)
            await asyncio.sleep(window)

            async with self._lock:
                if not self._buffer:
                    break  # Buffer boş, task sonlanır

            await self._execute_flush(force=False)

            async with self._lock:
                if not self._buffer:
                    break

    async def _execute_flush(self, force: bool = False) -> None:
        """
        Buffer'ı al, KEDA eşik kontrolü yap, mesajları gönder.
        force=True ise eşik kontrolü atlanır (graceful shutdown).
        """
        async with self._lock:
            if not self._buffer:
                return
            # Snapshot al ve buffer'ı temizle (atomik)
            snapshot = self._buffer.copy()
            self._buffer.clear()

        self._metrics["flushes_triggered"] += 1
        logger.info("📦 [FLUSH] %d sembol işleniyor...", len(snapshot))

        for symbol, acc in snapshot.items():
            await self._process_symbol(symbol, acc, force=force)

        logger.info("-" * 55)

    async def _process_symbol(
        self, symbol: str, acc: WelfordAccumulator, force: bool
    ) -> None:
        """
        Tek sembol için eşik kontrolü ve payload gönderimi.
        """
        tier = classify_cost_tier(acc.count, acc.max_change)

        # ── KEDA SUPPRESSION (Azure maliyet kalkanı) ──────────────────
        if not force and tier == "LOW":
            self._metrics["keda_suppressed"] += 1
            logger.info(
                "   🔇 SUPPRESSED %s | %d sinyal | Δ%%%.4f → Azure uyandırılmadı",
                symbol, acc.count, acc.max_change
            )
            return

        if not force and acc.count < self.min_signals_to_wake:
            self._metrics["keda_suppressed"] += 1
            logger.info(
                "   🔇 SUPPRESSED %s | sinyal sayısı yetersiz (%d < %d)",
                symbol, acc.count, self.min_signals_to_wake
            )
            return

        if not force and abs(acc.max_change) < self.min_change_pct_to_wake:
            self._metrics["keda_suppressed"] += 1
            logger.info(
                "   🔇 SUPPRESSED %s | değişim yetersiz (%.4f%% < %.4f%%)",
                symbol, acc.max_change, self.min_change_pct_to_wake
            )
            return

        # ── PAYLOAD ──────────────────────────────────────────────────
        payload = {
            "event": "aggregated_volatility",
            "symbol": symbol,
            "signal_count": acc.count,
            "avg_price": round(acc.mean, 6),
            "price_stddev": round(acc.stddev, 6),
            "price_range": round(acc.max_price - acc.min_price, 6),
            "max_change_pct": round(acc.max_change, 5),
            "cost_tier": tier,                         # KEDA filter key
            "window_start": acc.first_seen,
            "timestamp": time.time(),
        }

        await self._dispatch(payload)

    # ------------------------------------------------------------------
    # Dispatch (RabbitMQ + Redis fallback + circuit-breaker)
    # ------------------------------------------------------------------

    async def _dispatch(self, payload: dict) -> None:
        """
        Önce RabbitMQ dene. Circuit açıksa veya hata varsa Redis'e düş.
        """
        symbol = payload["symbol"]
        body = json.dumps(payload).encode()

        if not self._circuit_open and self.rabbitmq_channel is not None:
            try:
                await self.rabbitmq_channel.default_exchange.publish(
                    aio_pika.Message(
                        body=body,
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,  # mesaj kaybolmasın
                        content_type="application/json",
                    ),
                    routing_key=self.routing_key,
                )
                self._rabbitmq_errors = 0  # başarı → hata sayacı sıfırla
                self._metrics["keda_wakeups"] += 1
                logger.warning(
                    "🚀 KEDA [%s] %s | %d sinyal | tier=%s",
                    payload["cost_tier"], symbol,
                    payload["signal_count"], payload["cost_tier"]
                )
                return
            except Exception as exc:
                self._rabbitmq_errors += 1
                logger.error("RabbitMQ hata (%d/%d): %s",
                             self._rabbitmq_errors, self.circuit_breaker_limit, exc)
                if self._rabbitmq_errors >= self.circuit_breaker_limit:
                    self._circuit_open = True
                    self._metrics["circuit_breaker_trips"] += 1
                    logger.critical(
                        "⚡ CIRCUIT BREAKER AÇILDI — Redis fallback aktif"
                    )

        # ── Redis Fallback ────────────────────────────────────────────
        if self.redis is not None:
            try:
                await self.redis.publish(self.redis_fallback_channel, body)
                # Redis'te KEDA için list-based queue (KEDA redis scaler)
                await self.redis.rpush("vc:wakeup:queue", body)
                await self.redis.expire("vc:wakeup:queue", 300)  # 5 dk TTL
                self._metrics["keda_wakeups"] += 1
                logger.warning(
                    "📡 Redis FALLBACK [%s] %s | tier=%s",
                    payload["cost_tier"], symbol, payload["cost_tier"]
                )
            except Exception as exc:
                logger.critical("Redis fallback da başarısız: %s | Payload: %s", exc, payload)
        else:
            logger.critical(
                "HER İKİ TRANSPORT DOWN. Payload kaybedildi: %s", payload
            )

    # ------------------------------------------------------------------
    # Circuit-breaker manuel reset (ops/health endpoint'ten çağrılır)
    # ------------------------------------------------------------------

    def reset_circuit_breaker(self) -> None:
        self._circuit_open = False
        self._rabbitmq_errors = 0
        logger.info("Circuit breaker sıfırlandı.")

    def get_metrics(self) -> dict:
        return dict(self._metrics)
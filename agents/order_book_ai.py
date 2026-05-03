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
from typing import Deque, Dict, Optional
import ssl
import certifi
import random
import traceback
import aio_pika
import aiohttp
import redis.asyncio as redis
import websockets
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","agent":"OrderBookAI","level":"%(levelname)s","msg":%(message)s}',
)
logger = logging.getLogger("OrderBookAI")

@dataclass
class OrderBookConfig:
    symbol: str = field(default_factory=lambda: os.getenv("SYMBOL", "btcusdt"))
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://redis-service:6379/0"))
    rabbitmq_url: str = field(default_factory=lambda: os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq-service:5672/"))
    rabbitmq_queue: str = field(default_factory=lambda: os.getenv("RABBITMQ_QUEUE", "market_alerts"))
    imbalance_threshold: float = field(default_factory=lambda: float(os.getenv("IMBALANCE_THRESHOLD", "0.80")))
    depth_levels: int = field(default_factory=lambda: int(os.getenv("DEPTH_LEVELS", "15")))
    redis_write_interval_ms: int = field(default_factory=lambda: int(os.getenv("REDIS_WRITE_INTERVAL_MS", "1000")))
    ws_reconnect_max_backoff: int = field(default_factory=lambda: int(os.getenv("WS_RECONNECT_MAX_BACKOFF", "60")))
    metrics_port: int = field(default_factory=lambda: int(os.getenv("METRICS_PORT", "8010")))
    circuit_breaker_threshold: int = field(default_factory=lambda: int(os.getenv("CB_THRESHOLD", "5")))
    spread_spike_multiplier: float = field(default_factory=lambda: float(os.getenv("SPREAD_SPIKE_MULT", "3.0")))
    vwap_deviation_pct: float = field(default_factory=lambda: float(os.getenv("VWAP_DEV_PCT", "0.5")))
    trigger_cooldown_s: float = field(default_factory=lambda: float(os.getenv("TRIGGER_COOLDOWN_S", "5.0")))

class Metrics:
    ws_messages_total = Counter("orderbook_ws_messages_total", "Total WebSocket messages", ["symbol"])
    ws_reconnects_total = Counter("orderbook_ws_reconnects_total", "WebSocket reconnect count", ["symbol"])
    keda_triggers_total = Counter("orderbook_keda_triggers_total", "KEDA trigger count", ["symbol", "event_type"])
    redis_writes_total = Counter("orderbook_redis_writes_total", "Redis write count", ["symbol"])
    order_book_spread = Gauge("orderbook_spread", "Current bid-ask spread", ["symbol"])
    order_book_imbalance = Gauge("orderbook_imbalance", "Current order book imbalance", ["symbol"])
    order_book_bid_depth = Gauge("orderbook_bid_depth_usd", "Total bid depth USD", ["symbol"])
    order_book_ask_depth = Gauge("orderbook_ask_depth_usd", "Total ask depth USD", ["symbol"])
    vwap_deviation = Gauge("orderbook_vwap_deviation_pct", "VWAP deviation percent", ["symbol"])
    processing_latency = Histogram("orderbook_processing_latency_seconds", "Message processing latency", ["symbol"])
    circuit_breaker_state = Gauge("orderbook_circuit_breaker_open", "Circuit breaker state (1=open)", ["component"])

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    _failures: int = 0
    _last_failure_time: float = 0.0
    _state: CircuitState = CircuitState.CLOSED

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time > self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info(f'"circuit_breaker":"{self.name}","transition":"HALF_OPEN"')
        return self._state

    def record_success(self):
        self._failures = 0
        self._state = CircuitState.CLOSED
        Metrics.circuit_breaker_state.labels(component=self.name).set(0)

    def record_failure(self):
        self._failures += 1
        self._last_failure_time = time.monotonic()
        if self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            Metrics.circuit_breaker_state.labels(component=self.name).set(1)
            logger.error(f'"circuit_breaker":"{self.name}","transition":"OPEN","failures":{self._failures}')

    def allow_request(self) -> bool:
        return self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

@dataclass
class L2OrderBook:
    symbol: str
    bids: Dict[float, float] = field(default_factory=dict)
    asks: Dict[float, float] = field(default_factory=dict)
    last_update_id: int = 0
    is_synced: bool = False
    _spread_history: Deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def reset(self):
        self.bids.clear()
        self.asks.clear()
        self._spread_history.clear()
        self.last_update_id = 0
        self.is_synced = False
        logger.warning(f'"event":"L2_RESET","symbol":"{self.symbol}"')

    def apply_snapshot(self, snapshot: dict):
        self.bids = {float(p): float(q) for p, q in snapshot["bids"]}
        self.asks = {float(p): float(q) for p, q in snapshot["asks"]}
        self.last_update_id = snapshot["lastUpdateId"]
        self.is_synced = True
        logger.info(f'"event":"SNAPSHOT_APPLIED","lastUpdateId":{self.last_update_id},"bids":{len(self.bids)},"asks":{len(self.asks)}')

    def apply_delta(self, data: dict) -> bool:
        if not self.is_synced:
            return False

        U, u = data["U"], data["u"]

        if u < self.last_update_id + 1:
            return True

        if U > self.last_update_id + 1:
            logger.error(f'"event":"SEQUENCE_GAP","expected":{self.last_update_id+1},"got_U":{U}')
            self.is_synced = False
            return False

        self._update_side(data.get("b", []), self.bids)
        self._update_side(data.get("a", []), self.asks)
        self.last_update_id = u
        return True

    @staticmethod
    def _update_side(updates: list, book: Dict[float, float]):
        for price_str, qty_str in updates:
            price, qty = float(price_str), float(qty_str)
            if qty == 0.0:
                book.pop(price, None)
            else:
                book[price] = qty

    def top_bid(self) -> Optional[float]:
        return max(self.bids.keys()) if self.bids else None

    def top_ask(self) -> Optional[float]:
        return min(self.asks.keys()) if self.asks else None

    def spread(self) -> Optional[float]:
        b, a = self.top_bid(), self.top_ask()
        if b and a:
            s = a - b
            self._spread_history.append(s)
            return s
        return None

    def spread_avg(self) -> Optional[float]:
        if len(self._spread_history) < 10:
            return None
        return sum(self._spread_history) / len(self._spread_history)

    def compute_metrics(self, levels: int) -> Optional[dict]:
        if not self.bids or not self.asks:
            return None

        top_b = self.top_bid()
        top_a = self.top_ask()
        sp = self.spread()

        if top_b is None or top_a is None or sp is None:
            return None

        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:levels]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:levels]

        bid_vol = sum(q for _, q in sorted_bids)
        ask_vol = sum(q for _, q in sorted_asks)
        total_vol = bid_vol + ask_vol

        if total_vol == 0:
            return None

        imbalance = bid_vol / total_vol
        bid_vwap_num = sum(p * q for p, q in sorted_bids)
        ask_vwap_num = sum(p * q for p, q in sorted_asks)
        vwap = (bid_vwap_num + ask_vwap_num) / (bid_vol + ask_vol)
        mid = (top_b + top_a) / 2.0
        vwap_dev_pct = abs(vwap - mid) / mid * 100.0

        bid_depth_usd = sum(p * q for p, q in sorted_bids)
        ask_depth_usd = sum(p * q for p, q in sorted_asks)

        return {
            "top_bid": top_b,
            "top_ask": top_a,
            "spread": sp,
            "spread_avg": self.spread_avg(),
            "imbalance": imbalance,
            "bid_vol": bid_vol,
            "ask_vol": ask_vol,
            "bid_depth_usd": bid_depth_usd,
            "ask_depth_usd": ask_depth_usd,
            "vwap": vwap,
            "vwap_dev_pct": vwap_dev_pct,
            "mid_price": mid,
        }

class OrderBookAI:
    def __init__(self, config: Optional[OrderBookConfig] = None):
        self.config = config or OrderBookConfig()
        self.book = L2OrderBook(symbol=self.config.symbol)
        
        self.redis_client: Optional[redis.Redis] = None
        self.rabbitmq_conn: Optional[aio_pika.RobustConnection] = None
        self.rabbitmq_channel: Optional[aio_pika.Channel] = None

        self.cb_redis = CircuitBreaker("redis", failure_threshold=self.config.circuit_breaker_threshold)
        self.cb_rabbitmq = CircuitBreaker("rabbitmq", failure_threshold=self.config.circuit_breaker_threshold)

        self._last_redis_write: float = 0.0
        self.is_running = True

        self._analysis_event = asyncio.Event()

        self._snapshot_url = f"https://api.binance.com/api/v3/depth?symbol={self.config.symbol.upper()}&limit=1000"
        self._ws_url = f"wss://testnet.binance.vision/ws/{self.config.symbol.lower()}@depth20@1000ms"
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

    async def connect_infrastructure(self):
        logger.info('"event":"INFRA_CONNECT_START"')
        
        self.redis_client = redis.from_url(self.config.redis_url, decode_responses=True)
        await self.redis_client.ping()
        self.rabbitmq_conn = await aio_pika.connect_robust(
            self.config.rabbitmq_url,
            timeout=20
        )
        self.rabbitmq_channel = await self.rabbitmq_conn.channel()
        await self.rabbitmq_channel.set_qos(prefetch_count=100)
        await self.rabbitmq_channel.declare_queue(self.config.rabbitmq_queue, durable=True)
        logger.info('"event":"INFRA_CONNECT_OK"')

    async def _fetch_snapshot(self) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(self._snapshot_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def _trigger_keda(self, event_type: str, details: dict):
        if not self.cb_rabbitmq.allow_request():
            logger.warning(f'"event":"KEDA_SKIP","reason":"circuit_breaker_open","event_type":"{event_type}"')
            return

        payload = {
            "timestamp": time.time(),
            "event": event_type,
            "symbol": self.config.symbol,
            "details": details,
        }
        try:
            message = aio_pika.Message(
                body=json.dumps(payload).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )
            await self.rabbitmq_channel.default_exchange.publish(
                message, routing_key=self.config.rabbitmq_queue
            )
            self.cb_rabbitmq.record_success()
            Metrics.keda_triggers_total.labels(symbol=self.config.symbol, event_type=event_type).inc()
            logger.warning(f'"event":"KEDA_TRIGGERED","type":"{event_type}","details":{json.dumps(details)}')
        except Exception as exc:
            self.cb_rabbitmq.record_failure()
            logger.error(f'"event":"RABBITMQ_PUBLISH_FAIL","error":"{exc}"')

    async def _check_and_trigger(self, event_type: str, details: dict):
        lock_key = f"cooldown:master_trigger:{self.config.symbol}"
        acquired = await self.redis_client.set(
            lock_key, 
            f"locked_by_orderbook_{event_type}", # Kilit değerine açıklama ekle
            ex=int(self.config.trigger_cooldown_s), 
            nx=True
        )
        
        if not acquired:
            # Kilit zaten var. Master AI hala çalışıyor veya yeni uyandı. İşlemi atla (Drop).
            logger.debug(f'"event":"TRIGGER_DROPPED","reason":"cooldown_active","type":"{event_type}"')
            return
            
        # Kilit başarılıysa RabbitMQ'ya mesaj basarak Master AI'ı (KEDA'yı) uyandır
        await self._trigger_keda(event_type, details)

    async def _write_redis(self, metrics: dict):
        now = time.monotonic()
        interval = self.config.redis_write_interval_ms / 1000.0
        if now - self._last_redis_write < interval:
            return

        if not self.cb_redis.allow_request():
            return

        key = f"{self.config.symbol}_order_book"
        payload = {
            "top_bid": metrics["top_bid"],
            "top_ask": metrics["top_ask"],
            "spread": metrics["spread"],
            "spread_avg": metrics["spread_avg"],
            "imbalance": metrics["imbalance"],
            "bid_depth_usd": metrics["bid_depth_usd"],
            "ask_depth_usd": metrics["ask_depth_usd"],
            "vwap": metrics["vwap"],
            "vwap_dev_pct": metrics["vwap_dev_pct"],
            "mid_price": metrics["mid_price"],
            "ts": time.time(),
        }
        try:
            await self.redis_client.set(key, json.dumps(payload), ex=60)
            self._last_redis_write = now
            self.cb_redis.record_success()
            Metrics.redis_writes_total.labels(symbol=self.config.symbol).inc()
        except Exception as exc:
            self.cb_redis.record_failure()
            logger.error(f'"event":"REDIS_WRITE_FAIL","error":"{exc}"')

    async def _analyze_and_route(self):
        try:
            t0 = time.monotonic()
            metrics = await asyncio.to_thread(self.book.compute_metrics, self.config.depth_levels)
            if not metrics:
                return

            sym = self.config.symbol
            Metrics.order_book_spread.labels(symbol=sym).set(metrics["spread"])
            Metrics.order_book_imbalance.labels(symbol=sym).set(metrics["imbalance"])
            Metrics.order_book_bid_depth.labels(symbol=sym).set(metrics["bid_depth_usd"])
            Metrics.order_book_ask_depth.labels(symbol=sym).set(metrics["ask_depth_usd"])
            Metrics.vwap_deviation.labels(symbol=sym).set(metrics["vwap_dev_pct"])
            Metrics.processing_latency.labels(symbol=sym).observe(time.monotonic() - t0)

            await self._write_redis(metrics)

            imb = metrics["imbalance"]
            spread = metrics["spread"]
            spread_avg = metrics["spread_avg"]

            if imb > self.config.imbalance_threshold:
                await self._check_and_trigger("HEAVY_BUY_PRESSURE", {
                    "imbalance": round(imb, 4), 
                    "bid_depth_usd": round(metrics["bid_depth_usd"], 2), 
                    "mid_price": metrics["mid_price"]
                })
            elif imb < (1 - self.config.imbalance_threshold):
                await self._check_and_trigger("HEAVY_SELL_PRESSURE", {
                    "imbalance": round(imb, 4), 
                    "ask_depth_usd": round(metrics["ask_depth_usd"], 2), 
                    "mid_price": metrics["mid_price"]
                })
            if spread_avg and spread > spread_avg * self.config.spread_spike_multiplier:
                await self._check_and_trigger("SPREAD_SPIKE", {
                    "spread": round(spread, 6), 
                    "spread_avg": round(spread_avg, 6), 
                    "ratio": round(spread / spread_avg, 2),
                    "mid_price": metrics["mid_price"]
                })

            # 3. VWAP Deviation Tetikleyici
            if metrics["vwap_dev_pct"] > self.config.vwap_deviation_pct:
                await self._check_and_trigger("VWAP_DEVIATION", {
                    "vwap": round(metrics["vwap"], 2), 
                    "mid_price": round(metrics["mid_price"], 2),
                    "deviation_pct": round(metrics["vwap_dev_pct"], 4)
                })

        except Exception as exc:
            logger.error(f'"event":"ANALYSIS_ERROR","error":"{exc}"', exc_info=True)

    async def _analysis_worker(self):
        while self.is_running:
            await self._analysis_event.wait()
            self._analysis_event.clear()
            await self._analyze_and_route()
            await asyncio.sleep(0.001)
            
    async def _process_message(self, msg: dict) -> None:
        """Sentetik L2 verisini alıp orkestranın analiz aşamasını tetikler."""
        try:
            # Gelen mock veriyi mevcut defter (book) yapısına uydurmak istersen burayı genişletebilirsin.
            # Asıl kritik olan, verinin ardından analiz ve sinyal aşamasını (threshold kontrolünü) tetiklemektir:
            if hasattr(self, '_analyze_and_route'):
                await self._analyze_and_route()
        except Exception as e:
            logger.error(f'{{"event":"PROCESS_MESSAGE_ERROR","error":"{str(e)}"}}')

    async def stream_websocket(self):
        
        logger.info('{"event":"WS_MOCK_MODE","msg":"Fiziksel ağ engeli by-pass edildi. Sentetik L2 verisi üretiliyor."}')
        
        while self.is_running:
            self.book.reset()
            base_price += random.uniform(10.0, 60.0)  # Başlangıç fiyatı
            
            try:
                while self.is_running:
                    base_price += random.uniform(-15.0, 15.0)
                    mock_msg = {
                        "e": "depthUpdate",
                        "E": int(time.time() * 1000),
                        "s": self.config.symbol.upper(),
                        "U": random.randint(10000, 99999),
                        "u": random.randint(100000, 999999),
                        "b": [[str(round(base_price - i, 2)), str(round(random.uniform(0.1, 2.5), 3))] for i in range(1, 10)],
                        "a": [[str(round(base_price + i, 2)), str(round(random.uniform(0.1, 2.5), 3))] for i in range(1, 10)]
                    }
                    
                    await self._process_message(mock_msg)
                    
                    # Saniyede 2 mesaj (500ms frekans) ile sistemi besle
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                logger.error(f'{{"event":"MOCK_ERROR","error":"{str(e)}","trace":"{traceback.format_exc()}"}}')
                await asyncio.sleep(2)

    async def shutdown(self):
        logger.info('"event":"SHUTDOWN_START"')
        self.is_running = False
        await asyncio.sleep(0.5)
        if self.redis_client:
            await self.redis_client.aclose()
        if self.rabbitmq_conn:
            await self.rabbitmq_conn.close()
        logger.info('"event":"SHUTDOWN_COMPLETE"')

async def main():
    config = OrderBookConfig()
    start_http_server(config.metrics_port)
    logger.info(f'"event":"METRICS_SERVER_START","port":{config.metrics_port}')

    agent = OrderBookAI(config)
    
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(agent.shutdown()))

    await agent.connect_infrastructure()
    
    worker_task = asyncio.create_task(agent._analysis_worker())
    await agent.stream_websocket()
    worker_task.cancel()

if __name__ == "__main__":
    while True:
        try:
            asyncio.run(main())
        except Exception as e:
            logger.error(f'"event":"PROCESS_CRASHED","error":"{e}"')
            time.sleep(5)
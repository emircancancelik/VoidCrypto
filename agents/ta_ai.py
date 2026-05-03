from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import aio_pika
import aiohttp
import pandas as pd
import pandas_ta as ta
import redis.asyncio as redis

from typing import Optional
from pydantic import BaseModel, Field


logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}',
)
logger = logging.getLogger("TA_AI")

class TAConsensusPayload(BaseModel):
    symbol: str = Field(..., description="Trading pair, e.g. BTCUSDT")
    timeframe: str = Field(..., description="Candle interval, e.g. 15m")
    signal: int = Field(..., description="1 = Valid Entry, 0 = No Trade")
    confidence: float = Field(..., description="Layer approval ratio [0.0 – 1.0]")
    adx_value: float = Field(..., description="Current ADX — trend strength gate")
    active_layers: int = Field(..., description="Number of layers that approved signal")
    atr_volatility: float = Field(..., description="Raw ATR % — consumed by Risk AI for OCO/Trailing Stop sizing")


class MLFeatureVector(BaseModel):
    dist_ema9: float    # (close - EMA9) / close
    dist_ema21: float   # (close - EMA21) / close
    dist_ema50: float   # (close - EMA50) / close
    dist_ema200: float  # (close - EMA200) / close
    macd_hist_pct: float   # MACD histogram / close
    macd_line_pct: float   # MACD line / close
    atr_pct: float         # ATR / close  (same as atr_volatility in payload)
    rsi: float             # RSI [0–100], already dimensionless
    obv_roc: float         # OBV rate-of-change over 14 periods
    volume_roc: float      # Volume rate-of-change over 14 periods
    bb_width_pct: float    # (BBU - BBL) / BBM  — squeeze proxy
    adx: float             # ADX [0–100]
    vwap_dist: float       # (close - VWAP) / close
    msb_bullish: int       # 1 / 0 — Market Structure Break flag

class YahooFinanceAsyncProvider:
    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    BASE_URL = "https://query2.finance.yahoo.com/v8/finance/chart/"

    _RANGE_MAP: dict[str, str] = {
        "1m": "1d",
        "5m": "5d",
        "15m": "5d",
        "30m": "10d",
        "1h": "1mo",
        "1d": "1y",
    }

    @staticmethod
    def map_symbol(symbol: str) -> str:
        return symbol.replace("USDT", "-USD") if symbol.endswith("USDT") else symbol

    def determine_range(self, interval: str) -> str:
        return self._RANGE_MAP.get(interval, "1mo")

    async def fetch_ohlcv(self, symbol: str, interval: str = "15m") -> pd.DataFrame:
        yf_symbol = self.map_symbol(symbol)
        range_str = self.determine_range(interval)
        url = f"{self.BASE_URL}{yf_symbol}?interval={interval}&range={range_str}"

        try:
            async with aiohttp.ClientSession(headers=self.HEADERS) as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status != 200:
                        logger.warning(f'"event":"FETCH_FAILED","symbol":"{symbol}","http_status":{response.status}')
                        return pd.DataFrame()
                    raw = await response.json()
                    return self._parse(raw)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.error(f'"event":"FETCH_EXCEPTION","symbol":"{symbol}","error":"{exc}"')
            return pd.DataFrame()

    @staticmethod
    def _parse(data: dict) -> pd.DataFrame:
        try:
            result = data["chart"]["result"][0]
            quote = result["indicators"]["quote"][0]
            df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(result["timestamp"], unit="s"),
                    "open": quote["open"],
                    "high": quote["high"],
                    "low": quote["low"],
                    "close": quote["close"],
                    "volume": quote["volume"],
                }
            )
            df.dropna(subset=["close"], inplace=True)
            df.set_index("timestamp", inplace=True)
            df["volume"] = df["volume"].astype(float)
            return df
        except (KeyError, TypeError, IndexError):
            return pd.DataFrame()

class TechnicalAnalysisAgent:
    MIN_ROWS = 210  # EMA200 needs ≥200 rows; add buffer

    def __init__(
        self,
        adx_threshold: int = 20,
        min_layer_approval: int = 3,
        msb_window: int = 20,
    ) -> None:
        self.adx_threshold = adx_threshold
        self.min_layer_approval = min_layer_approval
        self.msb_window = msb_window

    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # --- Trend ---
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.adx(length=14, append=True)

        # --- Momentum ---
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)

        # --- Volatility ---
        df.ta.atr(length=14, append=True)
        df.ta.bbands(length=20, std=2, append=True)

        # --- Volume ---
        df.ta.vwap(append=True)
        df.ta.obv(append=True)
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()

        # --- OBV ROC (14-period) ---
        df["OBV_ROC_14"] = df["OBV"].pct_change(periods=14).fillna(0.0)

        # --- Volume ROC (14-period) ---
        df["VOLUME_ROC_14"] = df["volume"].pct_change(periods=14).fillna(0.0)

        # --- Market Structure Break ---
        df["swing_high"] = df["high"].rolling(window=self.msb_window).max()
        df["msb_bullish"] = (df["close"] > df["swing_high"].shift(1)).astype(int)

        return df

    def _build_feature_vector(self, bar: pd.Series) -> MLFeatureVector:
        close = bar["close"]

        def safe_pct(val: float, base: float) -> float:
            return round((val - base) / base, 6) if base != 0 else 0.0

        bb_width = 0.0
        bbu = bar.get("BBU_20_2.0", 0.0)
        bbm = bar.get("BBM_20_2.0", 0.0)
        bbl = bar.get("BBL_20_2.0", 0.0)
        if bbm and bbm != 0.0:
            bb_width = round((bbu - bbl) / bbm, 6)

        return MLFeatureVector(
            dist_ema9=safe_pct(bar.get("EMA_9", close), close),
            dist_ema21=safe_pct(bar.get("EMA_21", close), close),
            dist_ema50=safe_pct(bar.get("EMA_50", close), close),
            dist_ema200=safe_pct(bar.get("EMA_200", close), close),
            macd_hist_pct=safe_pct(bar.get("MACDh_12_26_9", 0.0) + close, close),
            macd_line_pct=safe_pct(bar.get("MACD_12_26_9", 0.0) + close, close),
            atr_pct=round(bar.get("ATRr_14", 0.0) / close, 6) if close else 0.0,
            rsi=round(bar.get("RSI_14", 50.0), 4),
            obv_roc=round(bar.get("OBV_ROC_14", 0.0), 6),
            volume_roc=round(bar.get("VOLUME_ROC_14", 0.0), 6),
            bb_width_pct=bb_width,
            adx=round(bar.get("ADX_14", 0.0), 4),
            vwap_dist=safe_pct(bar.get("VWAP_D", close), close),
            msb_bullish=int(bar.get("msb_bullish", 0)),
        )

    def _evaluate_long(self, bar: pd.Series) -> tuple[int, int, float]:
        adx_value = float(bar.get("ADX_14", 0.0))

        # Hard gate — no trend, no trade
        if adx_value < self.adx_threshold:
            return 0, 0, adx_value

        layer1 = bool(
            bar["EMA_9"] > bar["EMA_21"] and bar["close"] > bar["EMA_200"]
        )
        layer2 = bool(
            bar["RSI_14"] > 50 and bar["MACDh_12_26_9"] > 0
        )
        layer3 = bool(
            bar["close"] > bar["VWAP_D"]
            and bar["volume"] > bar["volume_sma_20"]
        )
        layer4 = bool(bar["msb_bullish"])

        active = int(layer1) + int(layer2) + int(layer3) + int(layer4)
        signal = 1 if active >= self.min_layer_approval else 0
        return signal, active, adx_value

    def process_sync(
        self, symbol: str, timeframe: str, df_raw: pd.DataFrame
    ) -> tuple[TAConsensusPayload, MLFeatureVector]:
        
        # --- EARLY-EXIT ---
        # ADX hesabı
        adx_df = df_raw.ta.adx(length=14)
        current_adx = float(adx_df.iloc[-1]['ADX_14']) if not adx_df.empty else 0.0

        # Eğer piyasa yataysa (Trend yoksa), ağır hesaplamaları atla
        if current_adx < self.adx_threshold:
            return self._build_empty_response(symbol, timeframe, current_adx)

        # --- 2. ADIM: AĞIR HESAPLAMALAR ---
        # Eğer buraya geldiysek piyasada trend vardır, tüm indikatörleri hesaplayabiliriz
        df = self._calculate(df_raw.copy())
        df.dropna(inplace=True)

        if df.empty or len(df) < 2:
            return self._build_empty_response(symbol, timeframe, current_adx)

        # Karar ve Feature Vector oluşturma
        bar = df.iloc[-1]
        signal, active_layers, adx_val = self._evaluate_long(bar)
        fv = self._build_feature_vector(bar)

        payload = TAConsensusPayload(
            symbol=symbol,
            timeframe=timeframe,
            signal=signal,
            confidence=round(active_layers / 4.0, 4) if signal == 1 else 0.0,
            adx_value=round(adx_val, 2),
            active_layers=active_layers,
            atr_volatility=fv.atr_pct,
        )
        return payload, fv

    def _build_empty_response(self, symbol: str, timeframe: str, adx: float) -> tuple[TAConsensusPayload, MLFeatureVector]:
        payload = TAConsensusPayload(
            symbol=symbol, timeframe=timeframe, signal=0, confidence=0.0,
            adx_value=round(adx, 2), active_layers=0, atr_volatility=0.0
        )
        fv = MLFeatureVector(
            dist_ema9=0.0, dist_ema21=0.0, dist_ema50=0.0, dist_ema200=0.0,
            macd_hist_pct=0.0, macd_line_pct=0.0, atr_pct=0.0, rsi=50.0,
            obv_roc=0.0, volume_roc=0.0, bb_width_pct=0.0, adx=round(adx, 2),
            vwap_dist=0.0, msb_bullish=0
        )
        return payload, fv

async def _connect_redis(url: str) -> redis.Redis:
    client = redis.from_url(url, decode_responses=True)
    await client.ping()
    logger.info('"event":"REDIS_CONNECTED"')
    return client


async def _connect_rabbitmq(url: str, queue: str) -> tuple[aio_pika.RobustConnection, aio_pika.Channel]:
    conn: aio_pika.RobustConnection = await aio_pika.connect_robust(url)
    channel = await conn.channel()
    await channel.set_qos(prefetch_count=10)
    await channel.declare_queue(queue, durable=True)
    logger.info(f'"event":"RABBITMQ_CONNECTED","queue":"{queue}"')
    return conn, channel

async def feature_engine_loop(symbol: str, timeframe: str) -> None:
    redis_url = os.environ["REDIS_URL"]
    rabbitmq_url = os.environ["RABBITMQ_URL"]
    rabbitmq_queue = os.environ.get("RABBITMQ_QUEUE", "market_alerts")
    trigger_cooldown_s = int(os.environ.get("TRIGGER_COOLDOWN_S", "60"))
    poll_interval_s = int(os.environ.get("POLL_INTERVAL_S", "15"))

    redis_client = await _connect_redis(redis_url)
    rmq_conn, rmq_channel = await _connect_rabbitmq(rabbitmq_url, rabbitmq_queue)

    data_provider = YahooFinanceAsyncProvider()
    agent = TechnicalAnalysisAgent(
        adx_threshold=int(os.environ.get("ADX_THRESHOLD", "20")),
        min_layer_approval=int(os.environ.get("MIN_LAYER_APPROVAL", "3")),
        msb_window=int(os.environ.get("MSB_WINDOW", "20")),
    )

    redis_key = f"model_features:{symbol.lower()}:{timeframe}"
    lock_key = f"cooldown:master_trigger:{symbol.lower()}"

    logger.info(f'"event":"LOOP_START","symbol":"{symbol}","timeframe":"{timeframe}"')

    while True:
        loop_start = time.monotonic()
        try:
            raw_df = await data_provider.fetch_ohlcv(symbol, timeframe)

            if raw_df.empty or len(raw_df) < agent.MIN_ROWS:
                logger.warning(
                    f'"event":"INSUFFICIENT_DATA","symbol":"{symbol}",'
                    f'"rows":{len(raw_df)},"required":{agent.MIN_ROWS}'
                )
            else:
                ta_payload, feature_vec = await asyncio.to_thread(
                    agent.process_sync, symbol, timeframe, raw_df
                )
                feature_dict = feature_vec.model_dump()
                feature_dict["ta_signal"] = ta_payload.signal
                feature_dict["ta_confidence"] = ta_payload.confidence
                feature_dict["ta_active_layers"] = ta_payload.active_layers
                feature_dict["ts"] = time.time()

                await redis_client.set(redis_key, json.dumps(feature_dict), ex=300)

                logger.info(
                    f'"event":"FEATURES_WRITTEN","symbol":"{symbol}",'
                    f'"signal":{ta_payload.signal},"adx":{ta_payload.adx_value},'
                    f'"layers":{ta_payload.active_layers},"confidence":{ta_payload.confidence}'
                )

                if ta_payload.signal == 1:
                    acquired = await redis_client.set(
                        lock_key,
                        "locked_by_ta_ai",
                        ex=trigger_cooldown_s,
                        nx=True, 
                    )

                    if acquired:
                        trigger_payload = {
                            "timestamp": time.time(),
                            "event": "KEDA_TRIGGERED",
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "type": "TA_STRONG_SIGNAL",
                            "details": {
                                "confidence": ta_payload.confidence,
                                "adx": ta_payload.adx_value,
                                "active_layers": ta_payload.active_layers,
                                "atr_pct": ta_payload.atr_volatility,
                                "redis_key": redis_key,
                            },
                        }
                        await rmq_channel.default_exchange.publish(
                            aio_pika.Message(
                                body=json.dumps(trigger_payload).encode(),
                                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                            ),
                            routing_key=rabbitmq_queue,
                        )
                        logger.warning(
                            f'"event":"KEDA_TRIGGERED","symbol":"{symbol}",'
                            f'"confidence":{ta_payload.confidence}'
                        )
                    else:
                        logger.debug(
                            f'"event":"TRIGGER_DROPPED","reason":"cooldown_active",'
                            f'"symbol":"{symbol}"'
                        )

        except redis.RedisError as exc:
            logger.error(f'"event":"REDIS_ERROR","error":"{exc}"')
            # Reconnect on next iteration; don't crash the loop
            try:
                redis_client = await _connect_redis(redis_url)
            except Exception:
                pass
        except aio_pika.exceptions.AMQPException as exc:
            logger.error(f'"event":"RABBITMQ_ERROR","error":"{exc}"')
            try:
                rmq_conn, rmq_channel = await _connect_rabbitmq(rabbitmq_url, rabbitmq_queue)
            except Exception:
                pass
        except Exception as exc:
            logger.error(f'"event":"LOOP_EXCEPTION","error":"{exc}"')

        elapsed = time.monotonic() - loop_start
        sleep_for = max(0.0, poll_interval_s - elapsed)
        await asyncio.sleep(sleep_for)

if __name__ == "__main__":
    _symbol = os.environ.get("SYMBOL", "BTCUSDT")
    _timeframe = os.environ.get("TIMEFRAME", "15m")

    try:
        asyncio.run(feature_engine_loop(_symbol, _timeframe))
    except KeyboardInterrupt:
        logger.info('"event":"SHUTDOWN","reason":"KeyboardInterrupt"')
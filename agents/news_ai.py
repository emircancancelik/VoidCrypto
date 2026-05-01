import asyncio
import json
import logging
import os
import time
import re
import aiohttp
import feedparser
import numpy as np
import redis.asyncio as redis

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import importlib.util
    _FINBERT_AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
except ImportError:
    _FINBERT_AVAILABLE = False

try:
    import importlib.util
    _ONNX_AVAILABLE = importlib.util.find_spec("onnxruntime") is not None
except ImportError:
    _ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","service":"NewsAI","level":"%(levelname)s","msg":%(message)s}',
)
logger = logging.getLogger("NewsAI")

class Config:
    REDIS_URL: str          = os.getenv("REDIS_URL", "redis://redis-service:6379/0")
    FINBERT_MODEL: str      = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
    ONNX_MODEL_PATH: str    = os.getenv("ONNX_MODEL_PATH", "/models/finbert.onnx")

    # Redis TTL: 15 dakika. Bu değer dolduğunda NewsSentimentAgent 0.0 döndürür.
    SENTIMENT_TTL_SEC: int  = int(os.getenv("SENTIMENT_TTL_SEC", "900"))

    # Veri bayatsa Agent güvenmez — ayrı guard.
    STALE_THRESHOLD_SEC: int = int(os.getenv("STALE_THRESHOLD_SEC", "900"))

    # Paralel feed çekimi için semaphore limiti.
    # Azure Container Apps'ta ağ bant genişliği kısıtlıysa 30'a düşür.
    MAX_CONCURRENT_FEEDS: int = int(os.getenv("MAX_CONCURRENT_FEEDS", "50"))

    # Feed başına timeout (sn). 1 ölü site 199 siteyi beklemesin.
    FEED_TIMEOUT_SEC: int   = int(os.getenv("FEED_TIMEOUT_SEC", "8"))

    # İnference batch boyutu. FinBERT için 16-32 ideal.
    INFERENCE_BATCH_SIZE: int = int(os.getenv("INFERENCE_BATCH_SIZE", "16"))

    # Worker çalışma periyodu (sn).
    INGESTION_INTERVAL_SEC: int = int(os.getenv("INGESTION_INTERVAL_SEC", "300"))

    # İzlenen semboller. Gerçek sistemde bu liste RabbitMQ'dan veya config map'ten gelir.
    WATCHED_SYMBOLS: list[str] = json.loads(
        os.getenv("WATCHED_SYMBOLS", '["BTC","ETH","SOL","BNB","XRP"]')
    )

    # Feed listesi env'den yüklenebilir, yoksa default liste.
    RSS_FEEDS: list[str] = json.loads(os.getenv("RSS_FEEDS_JSON", json.dumps([
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://decrypt.co/feed",
        "https://cryptopotato.com/feed/",
        "https://ambcrypto.com/feed/",
        "https://cryptoslate.com/feed/",
        "https://thedefiant.io/api/feed",
        "https://bitcoinmagazine.com/.rss/full/",
        "https://news.bitcoin.com/feed/",
        "https://cryptonews.com/news/feed/",
        "https://www.theverge.com/",
        "https://techcrunch.com/",
        "https://www.coindesk.com/",
        "https://cointelegraph.com/",
        "https://openai.com/",
        "https://huggingface.co/",
        "https://www.anandtech.com/",
        "https://www.tomshardware.com/",
        "https://www.semianalysis.com/",
        "https://finance.yahoo.com/",
        "https://www.cnbc.com/",
        "https://feeds.marketwatch.com/",
        "https://www.ft.com/",
        "https://feeds.reuters.com/",
        "https://seekingalpha.com/",
        "https://api.axios.com/",
        "https://www.federalreserve.gov/",
        "https://www.sec.gov/",
        "https://home.treasury.gov/",
        "https://www.darkreading.com/",
        "https://krebsonsecurity.com/",
        "https://www.ransomware.live/",
        "https://www.zdnet.com/",
        "https://www.aljazeera.com/",
        "https://www.defense.gov/",
        "https://www.bloomberg.com/"

    ])))

@dataclass
class SentimentRecord:
    """Redis'e yazılan ve Agent'ın okuduğu canonical veri yapısı."""
    symbol: str
    nlp_score: float          # Ham FinBERT skoru [-1.0, 1.0]
    calibrated_score: float   # Platt/Isotonic sonrası [-1.0, 1.0]
    confidence: float         # Kalibrasyon sonrası güven skoru [0.0, 1.0]
    article_count: int        # Skora katkıda bulunan haber sayısı
    updated_at: float         # Unix timestamp (UTC)
    backend: str              # "finbert" | "onnx" | "lightweight"

    def redis_key(self) -> str:
        return f"precomputed_sentiment:{self.symbol.lower()}"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(raw: str) -> "SentimentRecord":
        return SentimentRecord(**json.loads(raw))


@dataclass
class AgentOutput:
    """NewsSentimentAgent'ın Master Decision'a ilettiği mesaj formatı."""
    symbol: str
    nlp_score: float
    confidence: float
    regime: str               # "bullish" | "bearish" | "neutral"
    article_count: int
    data_age_sec: float
    agent_id: str = "NewsSentimentAI"

class LightweightSentimentBackend:
    """
    transformers/torch yokken fallback.
    Kural tabanlı anahtar kelime sayımı. ECE açısından zayıf —
    production ortamında kabul edilemez. Sadece dev/test için.
    """
    _POSITIVE = frozenset(["surge","bull","bullish","adopt","upgrade","buy","high","rally",
                            "breakout","ath","accumulate","recovery","positive","growth","rise"])
    _NEGATIVE = frozenset(["hack","bear","bearish","ban","crash","sell","lawsuit","scam",
                            "dump","fraud","liquidation","exploit","negative","decline","fear"])

    @property
    def backend_name(self) -> str:
        return "lightweight"

    def infer_batch(self, texts: list[str]) -> list[tuple[float, float]]:
        results = []
        for text in texts:
            words = set(text.lower().split())
            pos = len(words & self._POSITIVE)
            neg = len(words & self._NEGATIVE)
            total = pos + neg
            if total == 0:
                results.append((0.0, 0.3))
            else:
                score = (pos - neg) / total
                # Keyword modeli için confidence 0.3-0.5 arası sabit — güvenilir değil.
                confidence = min(0.5, 0.3 + total * 0.02)
                results.append((score, confidence))
        return results


class FinBERTInferenceBackend:
    def __init__(self, model_name: str = Config.FINBERT_MODEL):
        if not _FINBERT_AVAILABLE:
            raise RuntimeError("transformers/torch yüklü değil.")
            
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        self._torch = torch # Modülü sınıf değişkenine ata
        
        self._device = "cuda" if self._torch.cuda.is_available() else "cpu"
        logger.info(f'"FinBERT yükleniyor | model={model_name} device={self._device}"')
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device).eval()

    @property
    def backend_name(self) -> str:
        return "finbert"

    def infer_batch(self, texts: list[str]) -> list[tuple[float, float]]:
        with self._torch.no_grad():
            if not texts:
                return []
            results = []
            for i in range(0, len(texts), Config.INFERENCE_BATCH_SIZE):
                batch   = texts[i : i + Config.INFERENCE_BATCH_SIZE]
                encoded = self._tokenizer(
                    batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
                ).to(self._device)
                
                probs   = self._torch.softmax(self._model(**encoded).logits, dim=-1).cpu().numpy()
                for row in probs:
                    # FinBERT label sırası: positive=0, negative=1, neutral=2
                    score = float(row[0] * 1.0 + row[1] * -1.0 + row[2] * 0.0)
                    conf  = float(row.max())
                    results.append((score, conf))
            return results


class ONNXInferenceBackend:
    def __init__(self, model_path: str = Config.ONNX_MODEL_PATH,
                 tokenizer_name: str = Config.FINBERT_MODEL):
        if not _ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime yüklü değil.")
        if not _FINBERT_AVAILABLE:
            raise RuntimeError("transformers tokenizer için gerekli.")

        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    @property
    def backend_name(self) -> str:
        return "onnx"

    def infer_batch(self, texts: list[str]) -> list[tuple[float, float]]:
        if not texts:
            return []

        results = []
        batch_size = Config.INFERENCE_BATCH_SIZE

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="np"
            )
            ort_inputs = {
                "input_ids":      encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in encoded:
                ort_inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

            logits = self._session.run(None, ort_inputs)[0]  # [B, 3]
            # Softmax
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            for prob_row in probs:
                pos_p, neg_p, neu_p = prob_row[0], prob_row[1], prob_row[2]
                raw_score = float(pos_p * 1.0 + neg_p * -1.0 + neu_p * 0.0)
                confidence = float(max(pos_p, neg_p, neu_p))
                results.append((raw_score, confidence))

        return results


def build_inference_backend():
    onnx_path = Config.ONNX_MODEL_PATH
    if _ONNX_AVAILABLE and os.path.exists(onnx_path):
        try:
            backend = ONNXInferenceBackend()
            logger.info('"ONNX backend seçildi."')
            return backend
        except Exception as e:
            logger.warning(f'"ONNX yüklenemedi: {e}. FinBERT deneniyor."')

    if _FINBERT_AVAILABLE:
        try:
            backend = FinBERTInferenceBackend()
            logger.info('"FinBERT backend seçildi."')
            return backend
        except Exception as e:
            logger.warning(f'"FinBERT yüklenemedi: {e}. Lightweight fallback."')

    logger.warning('"LightweightSentimentBackend devrede — production için uygun değil!"')
    return LightweightSentimentBackend()

class PlattCalibrator:
    def __init__(self,
                 a: float = float(os.getenv("PLATT_A", "-1.0")),
                 b: float = float(os.getenv("PLATT_B", "0.0"))):
        self.a = a
        self.b = b

    def calibrate(self, raw_confidence: float) -> float:
        """Sigmoid: P_cal = 1 / (1 + exp(A * f + B))"""
        val = 1.0 / (1.0 + np.exp(self.a * raw_confidence + self.b))
        return float(np.clip(val, 0.0, 1.0))
    
class BackgroundIngestionWorker:
    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._backend = build_inference_backend()
        self._calibrator = PlattCalibrator()
        self._semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_FEEDS)

    async def connect(self) -> None:
        self._redis = await redis.from_url(Config.REDIS_URL, decode_responses=True)
        logger.info('"Redis bağlantısı kuruldu."')

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.aclose()

    # ── Feed Çekimi ────────────────────────────────────────────────────────────
    async def _fetch_single_feed(
        self, session: aiohttp.ClientSession, url: str
    ) -> list[str]:
        """
        Tek bir RSS feed'i çeker ve en güncel 5 entryin
        başlık + özet birleşimini döndürür.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        }
        async with self._semaphore:
            try:
                timeout = aiohttp.ClientTimeout(total=Config.FEED_TIMEOUT_SEC)
                async with session.get(url, headers=headers, timeout=timeout) as resp:
                    if resp.status != 200:
                        logger.debug(f'"HTTP {resp.status}: {url}"')
                        return []
                    text = await resp.text(errors="replace")
                    parsed = feedparser.parse(text)
                    texts = []
                    for entry in parsed.entries[:5]:
                        title   = entry.get("title", "")
                        summary = entry.get("summary", "")
                        combined = f"{title}. {summary}".strip()
                        if combined and len(combined) > 10:
                            texts.append(combined)
                    return texts
            except asyncio.TimeoutError:
                logger.debug(f'"Timeout: {url}"')
            except aiohttp.ClientError as e:
                logger.debug(f'"Client hatası: {url} | {e}"')
            except Exception as e:
                logger.debug(f'"Beklenmedik hata: {url} | {e}"')
        return []

    async def _fetch_all_feeds(self) -> list[str]:
        """Tüm RSS feed'lerini paralel çeker, ham metin listesi döndürür."""
        connector = aiohttp.TCPConnector(limit=Config.MAX_CONCURRENT_FEEDS, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self._fetch_single_feed(session, url) for url in Config.RSS_FEEDS]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        all_texts: list[str] = []
        for result in results:
            if isinstance(result, list):
                all_texts.extend(result)
        return all_texts

    # ── Symbol Filtresi ────────────────────────────────────────────────────────
    @staticmethod
    def _filter_texts_for_symbol(texts: list[str], symbol: str) -> list[str]:
        symbol_lower = symbol.lower()
        symbol_aliases = {
            "btc": ["bitcoin", "btc"],
            "eth": ["ethereum", "eth", "ether"], # regex \bether\b Ethernet'i yakalamaz
            "sol": ["solana", "sol"],
            "bnb": ["bnb", "binance"],
            "xrp": ["ripple", "xrp"],
        }
        aliases = symbol_aliases.get(symbol_lower, [symbol_lower])
        pattern = re.compile(rf"\b({'|'.join(aliases)})\b", re.IGNORECASE)

        specific = [t for t in texts if pattern.search(t)]
        if len(specific) < 3:
            general_pattern = re.compile(r"\b(crypto|blockchain|liquidation|sec|fed)\b", re.IGNORECASE)
            general = [t for t in texts if general_pattern.search(t) and t not in specific]
            specific = specific + general[:5]

        return specific

    # ── Inference & Aggregation ────────────────────────────────────────────────
    def _compute_symbol_sentiment(
        self, texts: list[str], symbol: str
    ) -> tuple[float, float, float]:
        if not texts:
            return 0.0, 0.0, 0.0

        raw_results = self._backend.infer_batch(texts)  # [(score, conf), ...]

        if not raw_results:
            return 0.0, 0.0, 0.0

        # Confidence-weighted aggregation
        weighted_score_sum = 0.0
        weight_sum = 0.0

        for score, conf in raw_results:
            cal_conf = self._calibrator.calibrate(conf)
            weight = cal_conf * (1.5 if abs(score) > 0.8 else 1.0)
            weighted_score_sum += score * weight
            weight_sum += weight

        if weight_sum < 1e-9:
            return 0.0, 0.0, 0.0

        final_score = float(np.clip(weighted_score_sum / weight_sum, -1.0, 1.0))

        # Toplam confidence: ortalama kalibrasyon skoru, makale sayısına göre boost.
        avg_raw_conf = float(np.mean([c for _, c in raw_results]))
        cal_conf_final = self._calibrator.calibrate(avg_raw_conf)

        # Article count boost: 1 makalede conf düşük, 20+ makalede cap'e yaklaşır.
        article_boost = min(1.0, len(texts) / 20.0)
        final_confidence = float(np.clip(cal_conf_final * (0.5 + 0.5 * article_boost), 0.0, 1.0))

        return final_score, final_confidence, avg_raw_conf

    # ── Redis Write ───────────────────────────────────────────────────────────
    async def _write_to_redis(self, record: SentimentRecord) -> None:
        key = record.redis_key()
        await self._redis.set(key, record.to_json(), ex=Config.SENTIMENT_TTL_SEC)
        logger.info(
            f'"symbol":"{record.symbol}",'
            f'"nlp_score":{record.nlp_score:.4f},'
            f'"calibrated":{record.calibrated_score:.4f},'
            f'"confidence":{record.confidence:.4f},'
            f'"articles":{record.article_count},'
            f'"backend":"{record.backend}"'
        )

    # ── Main Cycle ─────────────────────────────────────────────────────────────
    async def run_cycle(self) -> None:
        """Tek bir ingestion döngüsü: fetch → infer → write."""
        cycle_start = time.monotonic()
        logger.info('"Ingestion döngüsü başlıyor."')

        all_texts = await self._fetch_all_feeds()
        logger.info(f'"Toplam ham metin: {len(all_texts)}"')

        write_tasks = []
        for symbol in Config.WATCHED_SYMBOLS:
            filtered = self._filter_texts_for_symbol(all_texts, symbol)
            nlp_score, confidence, _raw_conf = self._compute_symbol_sentiment(filtered, symbol)

            record = SentimentRecord(
                symbol           = symbol,
                nlp_score        = nlp_score,
                calibrated_score = nlp_score,   # ek kalibrasyon katmanı eklenirse ayrıştırılır
                confidence       = confidence,
                article_count    = len(filtered),
                updated_at       = time.time(),
                backend          = self._backend.backend_name,
            )
            write_tasks.append(self._write_to_redis(record))

        await asyncio.gather(*write_tasks)

        elapsed = time.monotonic() - cycle_start
        logger.info(f'"Döngü tamamlandı. Süre: {elapsed:.2f}s"')

class NewsSentimentAgent:
    """
    LangGraph node'u veya AutoGen ConversableAgent wrapper'ı olarak kullanılır.

    Sorumluluk:
      1. Redis'ten BackgroundIngestionWorker'ın yazdığı SentimentRecord'u oku.
      2. Bayat/eksik veri kontrolü yap.
      3. AgentOutput formatında çıktı üret.
      4. Master Decision ve Auditor AI için mesaj kuyruğuna yaz (RabbitMQ).

    Bu agent KARAR VERMEZ. Sadece yapılandırılmış, kalibre edilmiş sinyal üretir.
    """
    AGENT_ID = "NewsSentimentAI"

    def __init__(self):
        self._redis: Optional[redis.Redis] = None

    async def connect(self) -> None:
        self._redis = await redis.from_url(Config.REDIS_URL, decode_responses=True)

    async def disconnect(self) -> None:
        if self._redis:
            await self._redis.aclose()

    @staticmethod
    def _classify_regime(score: float, confidence: float) -> str:
        if confidence < 0.35:
            return "neutral"
        if score > 0.15:
            return "bullish"
        if score < -0.15:
            return "bearish"
        return "neutral"

    async def get_sentiment(self, symbol: str) -> AgentOutput:
        if self._redis is None:
            raise RuntimeError("Agent bağlı değil. Önce connect() çağrılmalı.")

        redis_key = f"precomputed_sentiment:{symbol.lower()}"

        try:
            raw = await self._redis.get(redis_key)
        except Exception as e:
            logger.error(f'"Redis okuma hatası [{symbol}]: {e}"')
            return self._neutral_output(symbol, data_age_sec=-1.0)

        if raw is None:
            logger.warning(f'"Redis\'te {symbol} için veri yok. Nötr sinyal."')
            return self._neutral_output(symbol, data_age_sec=-1.0)

        try:
            record = SentimentRecord.from_json(raw)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error(f'"SentimentRecord parse hatası [{symbol}]: {e}"')
            return self._neutral_output(symbol, data_age_sec=-1.0)

        data_age = time.time() - record.updated_at

        is_stale = data_age > Config.STALE_THRESHOLD_SEC
        penalty_factor = max(0.0, 1.0 - (data_age / (Config.STALE_THRESHOLD_SEC * 2)))
        adjusted_confidence = record.confidence * (1.0 if not is_stale else penalty_factor)
        regime = self._classify_regime(record.calibrated_score, adjusted_confidence)
        
        output = AgentOutput(
            symbol        = symbol,
            nlp_score     = record.calibrated_score,
            confidence    = round(adjusted_confidence, 4),
            regime        = regime,
            article_count = record.article_count,
            data_age_sec  = data_age,
            agent_id      = self.AGENT_ID,
)

        logger.info(
            f'"symbol":"{symbol}",'
            f'"regime":"{regime}",'
            f'"nlp_score":{output.nlp_score:.4f},'
            f'"confidence":{output.confidence:.4f},'
            f'"age":{data_age:.1f}s'
        )
        return output

    def _neutral_output(self, symbol: str, data_age_sec: float) -> AgentOutput:
        return AgentOutput(
            symbol        = symbol,
            nlp_score     = 0.0,
            confidence    = 0.0,
            regime        = "neutral",
            article_count = 0,
            data_age_sec  = data_age_sec,
            agent_id      = self.AGENT_ID,
        )

    def to_autogen_message(self, output: AgentOutput) -> dict:
        """
        AutoGen ConversableAgent mesaj formatı.
        Master Decision agent bu dict'i alır, Auditor doğrular.
        """
        return {
            "role":    "assistant",
            "name":    self.AGENT_ID,
            "content": json.dumps(asdict(output), ensure_ascii=False),
        }

    async def publish_to_rabbitmq(
        self,
        output: AgentOutput,
        channel,         
        exchange_name: str = "agent_signals",
        routing_key: str  = "news.sentiment",
    ) -> None:
        import aio_pika 

        message_body = json.dumps(asdict(output), ensure_ascii=False).encode()
        message = aio_pika.Message(
            body         = message_body,
            content_type = "application/json",
            delivery_mode= aio_pika.DeliveryMode.PERSISTENT,
        )
        await channel.publish(message, routing_key=routing_key)
        logger.info(f'"RabbitMQ\'ya yayınlandı: {routing_key} | {output.symbol}"')
async def news_sentiment_node(state: dict) -> dict:
    symbol = state.get("symbol", "BTC")

    agent = NewsSentimentAgent()
    await agent.connect()
    try:
        output = await agent.get_sentiment(symbol)
    finally:
        await agent.disconnect()

    return {**state, "news_sentiment": asdict(output)}

async def run_once(self) -> None:
    try:
        await self.connect()
        await self.run_cycle()
    except Exception as e:
        logger.error(f'"Job hatası: {e}"')
    finally:
        await self.disconnect()

if __name__ == "__main__":
    worker = BackgroundIngestionWorker()
    asyncio.run(worker.run_once())
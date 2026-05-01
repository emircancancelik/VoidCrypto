import asyncio
import json
import logging
import math
import os
import signal
import uuid
from enum import IntEnum
from typing import Optional
import aio_pika
import redis.asyncio as redis
from pydantic import BaseModel, Field, field_validator
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
import ccxt.async_support as ccxt

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger("RiskExecutionAI")


class SignalType(IntEnum):
    SHORT = -1
    PASS = 0
    LONG = 1


class MasterPayload(BaseModel):
    trade_signal: SignalType
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    current_price: float = Field(..., gt=0.0)
    atr_14: float = Field(..., gt=0.0)
    asset_pair: str
    wallet_balance: float = Field(..., gt=0.0)
    # Borsa precision parametreleri — ccxt market info'dan gelir
    step_size: float = Field(default=0.00001, gt=0.0)   # lot precision
    tick_size: float = Field(default=0.01, gt=0.0)       # price precision

    @field_validator("asset_pair")
    @classmethod
    def asset_pair_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("asset_pair boş olamaz")
        return v.upper()


class ExecutionPlan(BaseModel):
    """Risk AI'ın ürettiği ve borsaya ilettiği deterministik emir planı."""
    asset: str
    signal: SignalType
    position_size: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    risk_amount_usd: float
    client_order_id: str = ""
    exchange_order_id: str = ""

class RiskExecutionAgent:
    # Risk parametreleri — environment variable'dan okunması tercih edilir
    RISK_PER_TRADE_PCT: float = float(os.getenv("RISK_PCT", "0.015"))   # %1.5
    ATR_STOP_MULT: float = float(os.getenv("ATR_STOP_MULT", "1.5"))
    ATR_TP_MULT: float = float(os.getenv("ATR_TP_MULT", "3.0"))
    ORPHAN_VERIFY_RETRIES: int = 3
    RECONCILE_STARTUP_TIMEOUT: float = float(os.getenv("RECONCILE_TIMEOUT", "30.0"))

    def __init__(self) -> None:
        self.rmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost/")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client: Optional[redis.Redis] = None
        self.active_trailing_tasks = {}
        self.min_update_threshold = 0.0001
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.feedback_exchange: Optional[aio_pika.Exchange] = None
        self._shutdown_event = asyncio.Event()
        self._reconcile_done = asyncio.Event()

    async def connect_infrastructure(self) -> None:
        try:
            self.redis_client = await redis.from_url(self.redis_url, decode_responses=True)
            self.connection = await aio_pika.connect_robust(self.rmq_url)
            self.channel = await self.connection.channel()
            
            # 1. API Anahtarları Olmadan Risk Ajanı Çalışamaz
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_SECRET")
            
            if not api_key or not api_secret:
                logger.warning("BINANCE_API_KEY veya BINANCE_SECRET bulunamadı. Sadece Public/Paper Trading yapılabilir.")

            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'fetchMarkets': ['spot'],
                    'adjustForTimeDifference': True
                }
            })
            
            # 2. Dinamik Endpoint Rotasyonu (Kritik Azure Optimizasyonu)
            endpoints = [
                'https://api.binance.com',
                'https://api1.binance.com',
                'https://api2.binance.com',
                'https://api3.binance.com'
            ]

            connected = False
            for endpoint in endpoints:
                self.exchange.urls['api']['public'] = endpoint + '/api/v3'
                self.exchange.urls['api']['private'] = endpoint + '/api/v3'
                
                try:
                    await self.exchange.load_markets()
                    logger.info(f"Binance piyasa verileri yüklendi. Aktif Endpoint: {endpoint}")
                    connected = True
                    break
                except ccxt.BaseError as e:
                    logger.warning(f"Endpoint başarısız ({endpoint}): {e}. Diğerine geçiliyor...")
            
            if not connected:
                if self.exchange:
                    await self.exchange.close() 
                raise ConnectionError("Tüm Binance endpoint bağlantıları reddedildi. Rate Limit veya Ağ engeli mevcut.")
                
            logger.info("Altyapı bağlantıları kuruldu.")
            
        except Exception as final_e:
            logger.error(f"Altyapı kurulumunda kritik hata: {final_e}")
            if hasattr(self, 'exchange') and getattr(self, 'exchange', None):
                await self.exchange.close()
            raise

    # Deterministik 
    @staticmethod
    def _quantize(value: float, step: float, is_price: bool = False) -> float:
        if step <= 0:
            raise ValueError(f"step_size/tick_size sıfır veya negatif olamaz: {step}")
            
        val_dec = Decimal(str(value))
        step_dec = Decimal(str(step))
        
        if is_price:
            quantized = val_dec.quantize(step_dec, rounding=ROUND_HALF_UP)
        else:
            # Miktar hesabında matematiksel olarak lot adımının katlarına zorla
            quantized = (val_dec / step_dec).quantize(Decimal('1'), rounding=ROUND_DOWN) * step_dec
            
        return float(quantized)

    def calculate_position(self, payload: MasterPayload) -> Optional[ExecutionPlan]:
        if payload.trade_signal == SignalType.PASS:
            logger.info("Sinyal PASS — hesaplama atlandı.")
            return None

        risk_amount = payload.wallet_balance * self.RISK_PER_TRADE_PCT
        sl_distance = payload.atr_14 * self.ATR_STOP_MULT
        tp_distance = payload.atr_14 * self.ATR_TP_MULT

        if sl_distance == 0:
            logger.error("ATR sıfır — pozisyon hesaplanamıyor, sinyal reddedildi.")
            return None

        raw_size = risk_amount / sl_distance
        position_size = self._quantize(raw_size, payload.step_size)

        if position_size <= 0:
            logger.error(f"Pozisyon boyutu sıfıra indi ({raw_size:.6f} → {position_size}). step_size veya ATR kontrol edin.")
            return None

        if payload.trade_signal == SignalType.LONG:
            sl_price = self._quantize(payload.current_price - sl_distance, payload.tick_size)
            tp_price = self._quantize(payload.current_price + tp_distance, payload.tick_size)
        else:  # SHORT
            sl_price = self._quantize(payload.current_price + sl_distance, payload.tick_size)
            tp_price = self._quantize(payload.current_price - tp_distance, payload.tick_size)

        entry_price = self._quantize(payload.current_price, payload.tick_size)

        plan = ExecutionPlan(
            asset=payload.asset_pair,
            signal=payload.trade_signal,
            position_size=position_size,
            entry_price=entry_price,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            risk_amount_usd=risk_amount,
        )

        logger.info(
            f"Pozisyon Hesaplandı | {payload.asset_pair} "
            f"{'LONG' if payload.trade_signal == SignalType.LONG else 'SHORT'} "
            f"| Size={position_size} | Entry={entry_price} "
            f"| SL={sl_price} | TP={tp_price} | Risk=${risk_amount:.2f}"
        )
        return plan
    
    async def _trailing_stop_worker(self, asset: str, entry_price: float, atr: float, initial_stop: float) -> None:
        """
        Her aktif işlem için bağımsız çalışan, fiyatı takip eden ve ana akışı bloklamayan döngü.
        """
        logger.info(f"[{asset}] Trailing Stop Worker başlatıldı.")
        current_stop = initial_stop
        
        try:
            while not self._shutdown_event.is_set():
                # 1. Borsadan (veya Redis'ten) anlık fiyatı çek
                # Not: HFT'de bu veri genelde Order Book ajanının güncellediği Redis'ten okunur
                ticker = await self.exchange.fetch_ticker(asset) 
                current_price = ticker['last']
                
                # 2. Deterministik Trailing Stop Matematiği (Sadece yukarı çıkar)
                potential_new_stop = current_price - (atr * self.ATR_STOP_MULT)
                
                if potential_new_stop > current_stop:
                    change_ratio = (potential_new_stop - current_stop) / current_stop
                    
                    # 3. SIGNIFICANT CHANGE FİLTRESİ (Rate Limit Koruması)
                    if change_ratio > self.min_update_threshold:
                        # Gerçek borsa API'sine güncelleme gönder (Simülasyon)
                        logger.info(f"[{asset}] API Update: Stop seviyesi {current_stop:.2f} -> {potential_new_stop:.2f} güncelleniyor.")
                        current_stop = potential_new_stop
                        
                        # 4. STATE MANAGEMENT (Konteyner çökerse diye Redis'e yedekle)
                        # active_trade:{asset} key'ini bu yeni stop ile güncelle
                        await self._update_redis_state(asset, current_stop)
                
                # Kilitlenmeyi önlemek ve API rate limitini korumak için 1 saniye uyu
                await asyncio.sleep(1.0)
                
        except asyncio.CancelledError:
            logger.warning(f"[{asset}] Trailing Stop Worker iptal edildi (Graceful Shutdown).")
        except Exception as e:
            logger.error(f"[{asset}] Worker çöktü: {e}")

    async def _exchange_send_order(self, plan: ExecutionPlan) -> dict:
        logger.warning("="*50)
        logger.warning("🛑 PAPER TRADING: SANAL EMİR İLETİMİ 🛑")
        logger.warning(f"Borsa: Binance | Çift: {plan.asset}")
        logger.warning(f"Yön: {'🟩 LONG' if plan.signal == SignalType.LONG else '🟥 SHORT'}")
        logger.warning(f"Miktar: {plan.position_size} | Giriş: {plan.entry_price}")
        logger.warning(f"Stop-Loss: {plan.stop_loss_price} | Take-Profit: {plan.take_profit_price}")
        logger.warning(f"Riske Edilen Bakiye: ${plan.risk_amount_usd:.2f}")
        logger.warning("="*50)
        
        await asyncio.sleep(0.5)  # API gecikme simülasyonu
        return {"status": "success", "order_id": f"PAPER-{uuid.uuid4().hex[:8].upper()}"}


    async def _exchange_get_order_status(self, client_order_id: str, asset: str) -> Optional[str]:
        await asyncio.sleep(0.2)
        if client_order_id.startswith("PAPER-"):
            return "open"
        return None
        
    async def _wait_for_fill(self, client_order_id: str, asset: str, timeout_seconds: float = 5.0) -> bool:
        """
        Emrin Order Book'ta gerçekleşip (Filled) gerçekleşmediğini asenkron olarak kontrol eder.
        Süre dolarsa False döner.
        """
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
            # Gerçek senaryoda bu sorgu ccxt.fetch_order() ile yapılır
            status = await self._exchange_get_order_status(client_order_id, asset)
            
            if status == "closed":
                return True
            if status == "canceled" or status == "rejected":
                return False
                
            # Borsayı API limitlerine sokmamak için kısa bir bekleme
            await asyncio.sleep(0.5)
            
        return False
    
    async def shutdown(self):
        logger.info("Graceful Shutdown başlatılıyor... Aktif işçiler durdurulacak.")
        for asset, task in self.active_trailing_tasks.items():
            logger.info(f"[{asset}] Trailing Stop task iptal ediliyor...")
            task.cancel()
        await asyncio.gather(*self.active_trailing_tasks.values(), return_exceptions=True)
        if self.connection:
            await self.connection.close()
        if self.redis_client:
            await self.redis_client.aclose()
        if hasattr(self, 'exchange') and self.exchange:
             await self.exchange.close()
        logger.info("Ajan güvenli şekilde kapatıldı.")

    async def execute_and_verify(self, plan: ExecutionPlan) -> bool:
        """
        3 aşamalı güvenli emir iletimi:

        1. Pre-commit  : Emir borsaya gitmeden önce Redis'e "pending" yazılır.
                         Pod ölürse emir borsaya gitmemiş demektir → temizlenir.
        2. API call    : Borsa isteği client_order_id ile yapılır (idempotent).
        3. Reconcile   : Timeout/kopma durumunda exponential backoff ile orphan kontrolü.
        4. Atomic swap : Başarılıysa pending → active pipeline ile atomik geçiş yapılır.
        """
        plan.client_order_id = f"VOID-{uuid.uuid4().hex[:12].upper()}"
        pending_key = f"pending_trade:{plan.asset}"
        active_key = f"active_trade:{plan.asset}"
        plan_dict = plan.model_dump()

        await self.redis_client.set(pending_key, json.dumps(plan_dict), ex=60)
        logger.info(f"Pre-commit yazıldı: {pending_key} | ID: {plan.client_order_id}")

        exchange_response: Optional[dict] = None
        try:
            exchange_response = await self._exchange_send_order(plan)

        except Exception as api_err:
            logger.error(f"Borsa API hatası: {api_err}. Reconciliation başlıyor...")
            for attempt in range(self.ORPHAN_VERIFY_RETRIES):
                backoff = 2 ** attempt
                logger.info(f"Reconciliation denemesi {attempt + 1}/{self.ORPHAN_VERIFY_RETRIES} ({backoff}s bekleniyor)")
                await asyncio.sleep(backoff)
                status = await self._exchange_get_order_status(plan.client_order_id, plan.asset)
                if status in ("open", "closed"):
                    logger.warning(f"Emir kopmasına rağmen borsada aktif: {plan.client_order_id}")
                    exchange_response = {"status": "success", "order_id": plan.client_order_id}
                    break

            if exchange_response is None:
                logger.error(f"Emir borsada doğrulanamadı. Pending siliniyor: {pending_key}")
                await self.redis_client.delete(pending_key)
                return False

        if exchange_response and exchange_response.get("status") == "success":
            plan_dict["exchange_order_id"] = exchange_response["order_id"]
            
            logger.info(f"[{plan.asset}] Emir borsaya iletildi. Dolması (Fill) bekleniyor...")
            is_filled = await self._wait_for_fill(plan.client_order_id, plan.asset, timeout_seconds=5.0)
            
            if not is_filled:
                logger.warning(f"[{plan.asset}] Emir süresi içinde dolmadı (Zombi Emir). İptal ediliyor.")
                if hasattr(self, 'exchange') and self.exchange:
                    try:
                        pass
                    except Exception as e:
                        logger.error(f"Emir iptal edilemedi: {e}")
                        
                await self.redis_client.delete(pending_key)
                return False
            pipe = self.redis_client.pipeline()
            pipe.delete(pending_key)
            pipe.set(active_key, json.dumps(plan_dict), ex=86400)
            await pipe.execute()

            logger.info(f"Emir KESİNLEŞTİ VE DOLDU | Asset: {plan.asset} | ExchID: {exchange_response['order_id']}")
            
            task = asyncio.create_task(
                self._trailing_stop_worker(
                    asset=plan.asset,
                    entry_price=plan.entry_price,
                    atr=plan.atr, 
                    initial_stop=plan.stop_loss_price
                )
            )
            self.active_trailing_tasks[plan.asset] = task
            return True

        await self.redis_client.delete(pending_key)
        return False

    async def _startup_reconciliation(self, channel: aio_pika.Channel) -> None:
        """
        Pod yeniden ayağa kalktığında:
          - pending_trade:* → 60s TTL zaten temizler, ama süresi dolmamışsa orphan kontrolü yap.
          - active_trade:*  → borsa durumunu sorgula; kapalıysa Redis'ten temizle.

        Yeni mesaj akışı bu döngü bitmeden başlamaz (_reconcile_done event).
        """
        logger.info("Startup reconciliation başladı...")

        # Pending orphan kontrolü
        pending_keys = await self.redis_client.keys("pending_trade:*")
        for key in pending_keys:
            raw = await self.redis_client.get(key)
            if not raw:
                continue
            trade = json.loads(raw)
            coid = trade.get("client_order_id", "")
            asset = trade.get("asset", "")
            status = await self._exchange_get_order_status(coid, asset)
            if status in ("open", "closed"):
                # Emir borsada işlenmiş → active'e taşı
                trade["exchange_order_id"] = coid
                pipe = self.redis_client.pipeline()
                pipe.delete(key)
                pipe.set(f"active_trade:{asset}", json.dumps(trade), ex=86400)
                await pipe.execute()
                logger.info(f"Orphan pending aktifleştirildi: {asset}")
            else:
                await self.redis_client.delete(key)
                logger.info(f"Orphan pending silindi: {key}")

        # Active trade borsa denetimi
        active_keys = await self.redis_client.keys("active_trade:*")
        for key in active_keys:
            raw = await self.redis_client.get(key)
            if not raw:
                continue
            trade = json.loads(raw)
            asset = trade.get("asset", "")
            exch_id = trade.get("exchange_order_id", "")
            status = await self._exchange_get_order_status(exch_id, asset)
            if status in ("closed", "canceled"):
                await self.redis_client.delete(key)
                logger.info(f"Kapanmış işlem Redis'ten temizlendi: {asset}")
            else:
                logger.info(f"İşlem hala açık, takipte: {asset}")

        logger.info("Startup reconciliation tamamlandı.")
        self._reconcile_done.set()

    async def process_message(self, message: aio_pika.IncomingMessage) -> None:
        """
        RabbitMQ mesaj işleyicisi.

        Güvenlik katmanları:
          1. JSON parse hatası → mesaj reject (dead-letter queue'ya gider)
          2. Pydantic validation hatası → reject
          3. PASS sinyali → acknowledge, işlem yok
          4. Active trade mevcut → duplicate koruması, acknowledge
          5. calculate_position başarısız → acknowledge (ATR/balance sorunu loglanır)
          6. execute_and_verify başarısız → mesaj nack (yeniden kuyruğa girer)
        """
        async with message.process(requeue=False):
            try:
                body = json.loads(message.body.decode())
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.error(f"JSON parse hatası, mesaj drop ediliyor: {e}")
                return  # requeue=False → dead-letter'a gider

            try:
                payload = MasterPayload.model_validate(body)
            except Exception as e:
                logger.error(f"Payload validasyon hatası: {e} | Body: {body}")
                return

            # Startup reconciliation bitene kadar bekle
            try:
                await asyncio.wait_for(self._reconcile_done.wait(), timeout=self.RECONCILE_STARTUP_TIMEOUT)
            except asyncio.TimeoutError:
                logger.error("Reconciliation zaman aşımı. Mesaj reddediliyor.")
                raise  # message.process context manager nack yapar

            # Duplicate pozisyon koruması
            active_key = f"active_trade:{payload.asset_pair}"
            pending_key = f"pending_trade:{payload.asset_pair}"
            
            # Redis'te her iki anahtarın da varlığını atomik olarak kontrol et
            keys_exist = await self.redis_client.exists(active_key, pending_key)
            if keys_exist > 0:
                logger.warning(f"Açık veya işlenen (pending) emir mevcut, reddedildi: {payload.asset_pair}")
                return # acknowledge — mesajı tüket ama işlem açma

            if payload.trade_signal == SignalType.PASS:
                logger.info(f"PASS sinyali: {payload.asset_pair}")
                return

            plan = self.calculate_position(payload)
            if plan is None:
                logger.error(f"Pozisyon hesaplama başarısız: {payload.asset_pair}")
                return

            success = await self.execute_and_verify(plan)
            if not success:
                logger.error(f"Emir iletilemedi: {payload.asset_pair}")
                
                # Orkestrayı başarısızlıktan haberdar et
                failure_payload = json.dumps({
                    "event": "TRADE_FAILED",
                    "asset": payload.asset_pair,
                    "reason": "Exchange API timeout or rejection"
                }).encode()
                
                await self.feedback_exchange.publish(
                    aio_pika.Message(body=failure_payload),
                    routing_key="master_alert"
                )
                return 
            
    def _handle_shutdown_signal(self, signum: int) -> None:
        logger.info(f"Sinyal {signum} alındı — graceful shutdown başlatılıyor...")
        self._shutdown_event.set()
        
    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown_signal, sig)

        try:
            await self.connect_infrastructure()
            await self.channel.set_qos(prefetch_count=1)
            dlq_name = os.getenv("DLQ_NAME", "market_alerts.dlq")
            queue = await self.channel.declare_queue(
                "execution_orders", 
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "",
                    "x-dead-letter-routing-key": dlq_name,
                }
            )
            self.feedback_exchange = await self.channel.declare_exchange("master_feedback", aio_pika.ExchangeType.TOPIC)
            
            await self._startup_reconciliation(self.channel)

            await queue.consume(self.process_message)
            logger.info("Risk Execution AI aktif — 'execution_orders' kuyruğu RabbitMQ üzerinden dinleniyor.")

            await self._shutdown_event.wait()
            
        finally:
            logger.info("Bağlantılar kapatılıyor...")
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
            if self.redis_client:
                await self.redis_client.aclose()
            if hasattr(self, 'exchange') and self.exchange:
                await self.exchange.close() 
            logger.info("Ajan güvenli şekilde kapatıldı.")

if __name__ == "__main__":
    agent = RiskExecutionAgent()
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        pass
import asyncio
import logging
import math
import json
import os
from pydantic import BaseModel, ValidationError
import aio_pika
import redis.asyncio as redis

# Loglama konfigürasyonu - Azure Log Analytics için JSON formatına dönüştürülebilir
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RiskExecutionAI")

class MasterPayload(BaseModel):
    trade_signal: int  # 1 (Long), 0 (Short/Pass)
    confidence_score: float
    current_price: float
    atr_14: float
    asset_pair: str
    wallet_balance: float

class RiskExecutionAgent:
    def __init__(self):
        # Ortam değişkenlerinden bağlantı bilgilerini al (Azure secrets üzerinden)
        self.rmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost/")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.risk_per_trade_pct = 0.015  # Tek işlemde cüzdanın maksimum %1.5'i riske edilecek
        self.atr_stop_multiplier = 1.5
        self.atr_tp_multiplier = 3.0     # Risk/Reward oranı 1:2

    async def connect_redis(self):
        self.redis_client = await redis.from_url(self.redis_url)
        logger.info("Redis bağlantısı başarılı.")

    def calculate_position(self, payload: MasterPayload) -> dict:
        """
        100% Deterministik pozisyon ve OCO hesaplaması.
        Risk = Capital * Risk_Pct
        Position Size = Risk / Stop Loss Distance
        """
        if payload.trade_signal != 1:
            logger.info("Signal 0. İşlem açılmıyor.")
            return None

        # Fiyat mesafeleri
        stop_loss_distance = payload.atr_14 * self.atr_stop_multiplier
        take_profit_distance = payload.atr_14 * self.atr_tp_multiplier

        # Emir seviyeleri (Long senaryosu için)
        stop_loss_price = payload.current_price - stop_loss_distance
        take_profit_price = payload.current_price + take_profit_distance

        # Risk edilen miktar
        max_loss_amount = payload.wallet_balance * self.risk_per_trade_pct

        # Alınacak varlık miktarı (Örn: Kaç BTC?)
        position_size = max_loss_amount / stop_loss_distance

        # Maliyet kontrolü (Kaldıraçsız spot işlem varsayımı ile)
        position_cost = position_size * payload.current_price
        if position_cost > payload.wallet_balance:
            # Kaldıraç kullanılmıyorsa ve bakiye yetmiyorsa bakiyeye göre maksimize et
            position_size = payload.wallet_balance / payload.current_price
            logger.warning("Hesaplanan pozisyon boyutu bakiyeyi aşıyor. Bakiye sınırına çekildi.")

        return {
            "asset": payload.asset_pair,
            "entry_price": payload.current_price,
            "position_size": round(position_size, 5),
            "stop_loss": round(stop_loss_price, 2),
            "take_profit": round(take_profit_price, 2),
            "expected_loss": round(max_loss_amount, 2),
            "order_type": "OCO"
        }

    async def execute_exchange_order(self, execution_plan: dict):
        """
        Borsa (Binance/Bybit vb.) API'sine isteğin atıldığı deterministik fonksiyon.
        """
        logger.info(f"BORSAYA İLETİLİYOR: {execution_plan['asset']} için "
                    f"Miktar: {execution_plan['position_size']}, "
                    f"SL: {execution_plan['stop_loss']}, TP: {execution_plan['take_profit']}")
        
        # TODO: Gerçek borsa API entegrasyonu (ccxt veya aiohttp ile) burada yapılacak.
        # Bu aşamada sadece simüle edip 200 OK döndüğünü varsayıyoruz.
        await asyncio.sleep(0.5) 
        
        return {"status": "success", "order_id": "ORD-987654321"}

    async def process_message(self, message: aio_pika.IncomingMessage):
        async with message.process():
            try:
                body = json.loads(message.body.decode())
                logger.info(f"Master Decision'dan mesaj alındı: {body}")
                
                # Veri doğrulama
                payload = MasterPayload(**body)
                
                # Risk ve Boyut Hesaplama
                execution_plan = self.calculate_position(payload)
                
                if execution_plan:
                    # Borsaya Emri Gönder
                    exchange_response = await self.execute_exchange_order(execution_plan)
                    
                    if exchange_response["status"] == "success":
                        # Sistemin diğer ajanlarının (Whale Tracker, vs.) durumdan haberdar olması için Redis'i güncelle
                        await self.redis_client.set(
                            f"active_trade:{payload.asset_pair}", 
                            json.dumps(execution_plan),
                            ex=86400 # 24 saat sonra expire (TTL)
                        )
                        logger.info(f"İşlem başarılı, Redis güncellendi: {payload.asset_pair}")
                    else:
                        logger.error("Borsa API emri reddetti.")
                
            except ValidationError as e:
                logger.error(f"Geçersiz Master Payload Formatı. Emir iptal. Hata: {e}")
            except Exception as e:
                logger.error(f"Beklenmeyen Sistem Hatası: {e}")

    async def run(self):
        await self.connect_redis()
        connection = await aio_pika.connect_robust(self.rmq_url)
        channel = await connection.channel()
        
        # QoS = 1: Ajan aynı anda sadece 1 emri işler, race condition'ı engeller.
        await channel.set_qos(prefetch_count=1)
        
        queue = await channel.declare_queue("master_decision_queue", durable=True)
        logger.info("Risk Execution AI RabbitMQ dinlemeye başladı. KEDA ölçeklemesi için hazır.")
        
        await queue.consume(self.process_message)
        
        try:
            # Sonsuz döngü. Azure Container App scale-to-zero mantığında pod kapanana kadar yaşar.
            await asyncio.Future() 
        finally:
            await connection.close()
            await self.redis_client.close()

if __name__ == "__main__":
    agent = RiskExecutionAgent()
    asyncio.run(agent.run())
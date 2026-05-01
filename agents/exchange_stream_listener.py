import asyncio
import json
import logging
import os
import signal
import ccxt.pro as ccxt_pro 
import aio_pika

logging.basicConfig(
    level=logging.INFO, 
    format='{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger("ExchangeStreamListener")

class ExchangeStreamListener:
    def __init__(self):
        self.rmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost/")
        self.exchange_id = os.getenv("EXCHANGE_ID", "binance") 
        self.api_key = os.getenv("API_KEY", "")
        self.secret = os.getenv("API_SECRET", "")
        self.connection = None
        self.channel = None
        self._shutdown_event = asyncio.Event()
        exchange_class = getattr(ccxt_pro, self.exchange_id)
        self.exchange = exchange_class({
            'apiKey': self.api_key,
            'secret': self.secret,
            'enableRateLimit': True,
        })

    async def connect_rabbitmq(self):
        self.connection = await aio_pika.connect_robust(self.rmq_url)
        self.channel = await self.connection.channel()
        self.target_exchange = await self.channel.declare_exchange(
            "master_wakeup_exchange", aio_pika.ExchangeType.DIRECT, durable=True
        )
        logger.info("RabbitMQ bağlantısı sağlandı. Wake-up tetikleyicisi hazır.")

    async def notify_master_ai(self, order_data: dict):
        payload = {
            "event": "ORDER_CLOSED",
            "asset": order_data['symbol'],
            "exchange_order_id": order_data['id'],
            "status": order_data['status'],
            "filled_amount": order_data['filled'],
            "timestamp": order_data['timestamp']
        }
        
        message = aio_pika.Message(
            body=json.dumps(payload).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        )
        
        await self.target_exchange.publish(message, routing_key="state_update")
        logger.info(f"Master AI KEDA Tetiklemesi Gönderildi: {payload['asset']} - {payload['status']}")

    async def watch_orders(self):
        """ 7/24 Borsa WebSocket'ini (User Data Stream) dinler. """
        logger.info(f"{self.exchange_id} WebSocket dinleniyor...")
        
        while not self._shutdown_event.is_set():
            try:
                orders = await self.exchange.watch_orders()
                
                for order in orders:
                    if order['status'] in ['closed', 'canceled']:
                        logger.info(f"Emir kapanışı tespit edildi: {order['symbol']} | Durum: {order['status']}")
                        await self.notify_master_ai(order)
                        
            except ccxt_pro.NetworkError as e:
                logger.warning(f"Ağ Hatası, yeniden bağlanılıyor... Detay: {e}")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Beklenmeyen WebSocket Hatası: {e}")
                await asyncio.sleep(5)

    def shutdown(self, signum, frame):
        logger.info(f"Kapanış sinyali alındı. Listener durduruluyor...")
        self._shutdown_event.set()

    async def run(self):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.shutdown, sig, None)

        await self.connect_rabbitmq()
        listen_task = asyncio.create_task(self.watch_orders())
        
        await self._shutdown_event.wait()
        
        listen_task.cancel()
        await self.exchange.close()
        if self.connection:
            await self.connection.close()
        logger.info("Stream Listener güvenli şekilde kapatıldı.")

if __name__ == "__main__":
    listener = ExchangeStreamListener()
    try:
        asyncio.run(listener.run())
    except KeyboardInterrupt:
        pass
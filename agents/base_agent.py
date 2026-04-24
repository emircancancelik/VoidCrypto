import asyncio
import os
import logging
import signal
from abc import ABC, abstractmethod
import redis.asyncio as redis

# Log formatı merkezi sistem için standardize edildi
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

class BaseVoidAgent(ABC):
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logging.getLogger(self.agent_name)
        
        # Azure/Local yapılandırması için Environment Variables
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = None
        self._is_running = True

    async def connect_infrastructure(self):
        """Bağlantı havuzu (connection pool) kullanarak Redis'e bağlanır."""
        try:
            # decode_responses=True production'da string işlemleri için kritiktir
            self.redis_client = redis.from_url(
                self.redis_url, 
                decode_responses=True,
                socket_keepalive=True,
                retry_on_timeout=True
            )
            # Bağlantıyı test et
            await self.redis_client.ping()
            self.logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            self.logger.error(f"Infrastructure connection failed: {e}")
            raise e

    @abstractmethod
    async def run(self):
        """Her ajan bu metodu override etmek zorundadır."""
        pass

    async def cleanup(self):
        """Konteyner kapanırken kaynakları serbest bırakır."""
        self._is_running = False
        if self.redis_client:
            await self.redis_client.close()
            self.logger.info(f"{self.agent_name} resources cleaned up.")

    async def start(self):
        """Yaşam döngüsü yönetimi ve Graceful Shutdown."""
        # Linux sinyallerini yakala (Azure Container Apps durdurma komutu için)
        loop = asyncio.get_running_loop()
        for s in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(s, lambda: asyncio.create_task(self.cleanup()))

        try:
            await self.connect_infrastructure()
            self.logger.info(f"{self.agent_name} starting main loop...")
            await self.run()
        except Exception as e:
            self.logger.critical(f"Fatal error in {self.agent_name}: {e}", exc_info=True)
        finally:
            await self.cleanup()

# --- ÖRNEK KULLANIM (Implementasyon) ---

class OrderBookAI(BaseVoidAgent):
    """Base class'tan türetilmiş gerçek bir ajan örneği."""
    async def run(self):
        # KEDA'nın 'scale-to-zero' mantığına uygun bir döngü
        while self._is_running:
            # İş mantığı buraya gelir
            data = await self.redis_client.get("market_threshold")
            self.logger.info(f"Current threshold: {data}")
            await asyncio.sleep(5) 

if __name__ == "__main__":
    # Ajanı ayağa kaldıracak giriş noktası
    agent = OrderBookAI("OrderBookAI")
    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        pass
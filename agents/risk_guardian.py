import time
import json
import asyncio
import logging
from typing import Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)

class DeterministicRiskGuard:
    """
    Master AI'dan gelen olasılıksal kararları, sabit matematiksel eşiklerle
    denetleyen nihai karar (Execution) filtresidir. 
    Yapay zeka veya LLM çıkarımı içermez.
    """
    def __init__(self, max_risk_pct: float = 0.02, max_atr_multiplier: float = 3.0, max_latency_ms: int = 500):
        self.max_risk_pct = max_risk_pct          # İşlem başına maksimum kasa riski (Örn: %2)
        self.max_atr_multiplier = max_atr_multiplier # Anormal volatilite filtresi
        self.max_latency_ms = max_latency_ms      # Sinyal gecikme toleransı (Slippage koruması)
        self.logger = logging.getLogger("RiskGuard")

    def _check_latency(self, signal_timestamp: float) -> bool:
        """Sinyalin üretildiği an ile işlendiği an arasındaki gecikmeyi ölçer."""
        current_time = time.time() * 1000 # ms
        latency = current_time - signal_timestamp
        if latency > self.max_latency_ms:
            self.logger.warning(f"REJECTED: Sinyal gecikmesi çok yüksek ({latency:.2f}ms > {self.max_latency_ms}ms).")
            return False
        return True

    def _check_volatility(self, current_atr: float, baseline_atr: float) -> bool:
        """Piyasanın o anki volatilitesinin güvenli sınırlar içinde olup olmadığını denetler."""
        if current_atr > (baseline_atr * self.max_atr_multiplier):
            self.logger.warning(f"REJECTED: Aşırı volatilite tespit edildi. Mevcut ATR: {current_atr}")
            return False
        return True

    def _calculate_position_size(self, signal_price: float, stop_loss: float, wallet_balance: float) -> float:
        """
        Matematiksel kesinlikle açılacak maksimum pozisyon büyüklüğünü hesaplar.
        Master AI ne derse desin, risk limitini aşamaz.
        """
        risk_amount = wallet_balance * self.max_risk_pct
        price_risk_per_unit = abs(signal_price - stop_loss)
        
        if price_risk_per_unit <= 0:
            return 0.0
            
        allowed_position_size = risk_amount / price_risk_per_unit
        return allowed_position_size

    async def evaluate_and_route(self, master_payload: str, market_context: Dict) -> Optional[Dict]:
        """
        RabbitMQ'dan gelen Master AI payload'unu alır ve işler.
        Tüm koşullar sağlanırsa Execution AI'a gönderilecek nihai emri döner.
        """
        try:
            payload = json.loads(master_payload)
            action = payload.get("action")
            signal_price = float(payload.get("price", 0.0))
            stop_loss = float(payload.get("stop_loss", 0.0))
            signal_ts = float(payload.get("timestamp", 0.0))
            
            wallet_balance = float(market_context.get("available_balance", 0.0))
            current_atr = float(market_context.get("current_atr", 1.0))
            baseline_atr = float(market_context.get("baseline_atr", 1.0))

        except (ValueError, KeyError, TypeError) as e:
            self.logger.error(f"REJECTED: Hatalı veya eksik Master AI Payload'u. Hata: {e}")
            return None

        # 1. Gecikme (Latency) Kontrolü
        if not self._check_latency(signal_ts):
            return None

        # 2. Volatilite Kontrolü
        if not self._check_volatility(current_atr, baseline_atr):
            return None

        # 3. Yön ve Mantık Kontrolü (Hard-Logic Anomaly Detection)
        if action not in ["BUY", "SELL"]:
            self.logger.error(f"REJECTED: Geçersiz işlem yönü ({action}).")
            return None
            
        if (action == "BUY" and stop_loss >= signal_price) or (action == "SELL" and stop_loss <= signal_price):
            self.logger.error("REJECTED: Stop-Loss seviyesi işlem yönüyle mantıksal olarak çelişiyor.")
            return None

        # 4. Pozisyon Büyüklüğü (Risk) Hesaplama
        safe_position_size = self._calculate_position_size(signal_price, stop_loss, wallet_balance)
        
        if safe_position_size <= 0:
            self.logger.warning("REJECTED: Hesaplanabilir güvenli bir pozisyon boyutu bulunamadı.")
            return None

        # Onaylanmış Deterministik Emir Paketi
        approved_order = {
            "action": action,
            "order_type": "LIMIT",
            "price": signal_price,
            "quantity": safe_position_size,
            "stop_loss": stop_loss,
            "take_profit": payload.get("take_profit"), # Master AI'ın hedefini kabul edebiliriz, riski biz yönettik
            "execution_timestamp": time.time() * 1000
        }
        
        self.logger.info(f"APPROVED: Emir doğrulandı ve Execution katmanına iletiliyor -> {approved_order}")
        return approved_order

# --- RabbitMQ / KEDA Tüketici (Consumer) Entegrasyon Örneği ---
async def consume_master_queue(redis_client, rabbit_channel):
    """
    Sistemin asenkron yapısına entegre edilmiş tüketici fonksiyonu.
    Scale-to-zero mimarisinde sistem uyandığında çalışacak ana döngü.
    """
    risk_guard = DeterministicRiskGuard()
    
    # Mock edilmiş veri (Gerçek sistemde RabbitMQ queue'dan ve Redis'ten okunur)
    master_ai_message = json.dumps({
        "action": "BUY",
        "price": 65000.0,
        "stop_loss": 63000.0,
        "take_profit": 70000.0,
        "timestamp": time.time() * 1000 # Gecikmeyi simüle etmek için bunu değiştirebilirsin
    })
    
    market_context_from_redis = {
        "available_balance": 10000.0,
        "current_atr": 1500.0,
        "baseline_atr": 1400.0
    }
    
    # Sinyali İşle
    final_execution_payload = await risk_guard.evaluate_and_route(
        master_ai_message, 
        market_context_from_redis
    )
    
    if final_execution_payload:
        # Burada doğrudan borsa API'sine (Execution AI) asenkron istek atılır
        pass
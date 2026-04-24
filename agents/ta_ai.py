import asyncio
import aiohttp
import pandas as pd
import pandas_ta as ta
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# 1. PAYLOAD MİMARİSİ (Master AI & Risk AI Entegrasyonu)
# ---------------------------------------------------------
class TAConsensusPayload(BaseModel):
    symbol: str = Field(..., description="İşlem çifti, örn: BTCUSDT")
    timeframe: str = Field(..., description="Zaman dilimi, örn: 15m")
    signal: int = Field(..., description="1: Valid Entry (Long/Short), 0: No Trade")
    confidence: float = Field(..., description="0.0 - 1.0 arası katman onay skoru")
    adx_value: float = Field(..., description="Trendin mevcut gücü")
    active_layers: int = Field(..., description="Onay veren katman sayısı")
    atr_volatility: float = Field(..., description="Risk AI'ın deterministik OCO/Trailing Stop hesabı için ham metrik")

# ---------------------------------------------------------
# 2. ASYNC YAHOO FINANCE DATA PROVIDER (I/O Optimization)
# ---------------------------------------------------------
class YahooFinanceAsyncProvider:
    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    BASE_URL = "https://query2.finance.yahoo.com/v8/finance/chart/"

    @staticmethod
    def map_symbol(symbol: str) -> str:
        if symbol.endswith("USDT"):
            return symbol.replace("USDT", "-USD")
        return symbol

    @staticmethod
    def determine_range(interval: str) -> str:
        # EMA 200 ve Market Structure geriye dönük hesaplamaları için güvenli minimum data aralıkları
        range_map = {
            "1m": "1d",
            "5m": "5d",
            "15m": "5d",
            "30m": "10d",
            "1h": "1mo",
            "1d": "1y"
        }
        return range_map.get(interval, "1mo")

    async def fetch_ohlcv(self, symbol: str, interval: str = "15m") -> pd.DataFrame:
        yf_symbol = self.map_symbol(symbol)
        range_str = self.determine_range(interval)
        url = f"{self.BASE_URL}{yf_symbol}?interval={interval}&range={range_str}"

        async with aiohttp.ClientSession(headers=self.HEADERS) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return pd.DataFrame()
                data = await response.json()
                return self._parse_yf_response(data)

    def _parse_yf_response(self, data: dict) -> pd.DataFrame:
        try:
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            indicators = result['indicators']['quote'][0]

            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps, unit='s'),
                'open': indicators['open'],
                'high': indicators['high'],
                'low': indicators['low'],
                'close': indicators['close'],
                'volume': indicators['volume']
            })
            
            df.dropna(subset=['close'], inplace=True)
            df.set_index('timestamp', inplace=True)
            df['volume'] = df['volume'].astype(float)
            return df
        except (KeyError, TypeError, IndexError):
            return pd.DataFrame()

# ---------------------------------------------------------
# 3. TECHNICAL ANALYSIS AGENT CORE (Vectorized Logic)
# ---------------------------------------------------------
class TechnicalAnalysisAgent:
    def __init__(self, adx_threshold: int = 20, min_layer_approval: int = 3, msb_window: int = 20):
        self.adx_threshold = adx_threshold
        self.min_layer_approval = min_layer_approval
        self.msb_window = msb_window

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Trend (Katman 1 & Hard Constraint)
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.adx(length=14, append=True)

        # 2. Momentum (Katman 2)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)

        # 3. Volatilite (Risk AI Payload & Ek Bağlam)
        df.ta.atr(length=14, append=True)
        df.ta.bbands(length=20, std=2, append=True)

        # 4. Hacim Doğrulaması (Katman 3)
        df.ta.vwap(append=True)
        # Hacim ortalaması hesaplama (Volume spike tespiti için)
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()

        # 5. Basic Market Structure (Katman 4 - Vektörize Zirve/Dip ve MSB)
        df['swing_high'] = df['high'].rolling(window=self.msb_window).max()
        # Market Structure Break (BOS/MSB): Kapanışın bir önceki dönemin lokal zirvesini geçmesi
        df['msb_bullish'] = df['close'] > df['swing_high'].shift(1)

        return df

    def evaluate_long_logic(self, current_bar: pd.Series) -> tuple[int, int, float]:
        """Sıfır Machine Learning, %100 Kurallı Karar Ağacı"""
        
        # Hard Constraint: ADX Filtresi
        adx_value = current_bar.get('ADX_14', 0)
        if adx_value < self.adx_threshold:
            return 0, 0, adx_value 

        # Katman 1: Trend Onayı
        layer1_trend = (current_bar['EMA_9'] > current_bar['EMA_21']) and (current_bar['close'] > current_bar['EMA_200'])

        # Katman 2: Momentum Onayı
        layer2_momentum = (current_bar['RSI_14'] > 50) and (current_bar['MACDh_12_26_9'] > 0)

        # Katman 3: Hacim & Kurumsal İz Onayı
        # Fiyat VWAP'ın üstünde ve mevcut mumun hacmi 20 periyotluk hacim ortalamasından yüksek
        layer3_volume = (current_bar['close'] > current_bar['VWAP_D']) and (current_bar['volume'] > current_bar['volume_sma_20'])

        # Katman 4: Basic Market Structure
        # Fiyat, geçmiş 20 periyodun lokal zirvesini kırmış olmalı (MSB/BOS)
        layer4_structure = current_bar['msb_bullish']

        active_layers = sum([layer1_trend, layer2_momentum, layer3_volume, layer4_structure])

        # En az N katman onayı
        signal = 1 if active_layers >= self.min_layer_approval else 0

        return signal, active_layers, adx_value

    async def process_market_data(self, symbol: str, timeframe: str, df_ohlcv: pd.DataFrame) -> TAConsensusPayload:
        df = self.calculate_indicators(df_ohlcv)
        df.dropna(inplace=True)

        if df.empty or len(df) < 2:
            return TAConsensusPayload(
                symbol=symbol, timeframe=timeframe, signal=0, confidence=0.0, 
                adx_value=0.0, active_layers=0, atr_volatility=0.0
            )

        current_bar = df.iloc[-1]
        signal, active_layers, adx_val = self.evaluate_long_logic(current_bar)
        
        confidence = (active_layers / 4.0) if signal == 1 else 0.0

        return TAConsensusPayload(
            symbol=symbol,
            timeframe=timeframe,
            signal=signal,
            confidence=confidence,
            adx_value=round(adx_val, 2),
            active_layers=active_layers,
            atr_volatility=round(current_bar.get('ATRr_14', 0.0), 4)
        )

# ---------------------------------------------------------
# 4. KEDA / RABBITMQ WORKER ENTRY POINT
# ---------------------------------------------------------
async def ta_agent_worker(rabbitmq_message: dict) -> dict:
    """
    Azure Container App instance'ı ayağa kalktığında tetiklenen ana fonksiyon.
    1. Veriyi çeker (I/O).
    2. Modeli çalıştırır (CPU).
    3. JSON'a dönüştürür ve Master Decision AI kuyruğuna iletmek üzere döner.
    """
    symbol = rabbitmq_message.get("symbol")
    timeframe = rabbitmq_message.get("timeframe")
    
    # 1. Asenkron Data Fetch
    data_provider = YahooFinanceAsyncProvider()
    raw_df = await data_provider.fetch_ohlcv(symbol, timeframe)
    
    if raw_df.empty or len(raw_df) < 200:
        return {"error": "Insufficient data for EMA 200 or API limits hit.", "symbol": symbol}
    
    # 2. Vektörize Hesaplama ve Karar Mekanizması
    agent = TechnicalAnalysisAgent(adx_threshold=20, min_layer_approval=3, msb_window=20)
    ta_result_payload = await agent.process_market_data(symbol, timeframe, raw_df)
    
    # 3. Master Decision AI'a İletim Aşaması (Örn: aio_pika ile RabbitMQ Publish)
    # await mq_channel.default_exchange.publish(
    #    aio_pika.Message(body=ta_result_payload.model_dump_json().encode()),
    #    routing_key="master_ai_consensus_queue"
    # )
    
    return ta_result_payload.model_dump()

if __name__ == "__main__":
    # Test için RabbitMQ'dan geliyormuş gibi sahte bir mesaj oluşturuyoruz
    test_msg = {"symbol": "BTCUSDT", "timeframe": "15m"}
    
    # Asenkron fonksiyonu ayağa kaldırıyoruz
    result = asyncio.run(ta_agent_worker(test_msg))
    
    # Master AI'a gidecek olan nihai JSON payload'u ekrana yazdırıyoruz
    import json
    print(json.dumps(result, indent=4))
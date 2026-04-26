import asyncio
import json
import time
import websockets
import redis.asyncio as redis  # KRİTİK: Senkron redis yerine asenkron redis kullanıyoruz

BINANCE_URL = "wss://stream.binance.com:9443/ws/btcusdt@bookTicker"

async def fetch_data():
    # Asenkron Redis bağlantısı
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    # Bağlantıyı test et (Hata varsa baştan patlasın, sessizce beklemesin)
    try:
        await r.ping()
        print("[*] Asenkron Redis bağlantısı başarılı.")
    except Exception as e:
        print(f"[!] Redis bağlantı hatası: {e}")
        return

    while True: 
        try:
            async with websockets.connect(BINANCE_URL) as websocket:
                print("[*] Binance BookTicker WebSocket bağlandı.")
                
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    # RiskGuard'ın gecikmeyi (latency) hesaplayabilmesi için 
                    # sistemin veriyi aldığı anın zaman damgasını ekliyoruz.
                    clean_data = {
                        "symbol": data["s"],
                        "bid_price": float(data["b"]),
                        "ask_price": float(data["a"]),
                        "local_timestamp": time.time() * 1000  # Milisaniye
                    }
                    
                    # JSON'a çevir ve değişkene ATA
                    payload = json.dumps(clean_data)
                    
                    # 1. State Tutma: Sistemin her an son fiyata ulaşabilmesi için SET et
                    await r.set(f"market_context:{data['s'].lower()}:latest", payload)
                    
                    # 2. Event Fırlatma: KEDA'yı tetikleyecek / Ajanları uyandıracak mesaj
                    await r.publish("market_events:bookTicker", payload)
                    
                    # NOT: Prod ortamında bu print satırlarını silmelisin. 
                    # Terminale I/O (yazdırma) işlemi yapmak HFT sistemlerinde gereksiz gecikme yaratır.
                    # print(f"Published -> {clean_data['symbol']}: Bid {clean_data['bid_price']} | Ask {clean_data['ask_price']}")
                    
        except websockets.exceptions.ConnectionClosedError:
            print("[!] Binance WebSocket bağlantısı koptu. Yeniden bağlanılıyor...")
            await asyncio.sleep(1)
        except Exception as e:
            print(f"[!] Beklenmeyen Hata: {e}")
            await asyncio.sleep(2)

if __name__ == "__main__":
    try:
        asyncio.run(fetch_data())
    except KeyboardInterrupt:
        print("\n[*] Veri akışı kullanıcı tarafından durduruldu.")
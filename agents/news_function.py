import azure.functions as func
import logging
import os
import json
import asyncio
import feedparser
import aiohttp
from datetime import datetime, timezone
import redis.asyncio as redis

app = func.FunctionApp()

REDIS_URL = os.getenv("REDIS_URL", "redis://redis-service:6379/0")

async def process_rss_and_score():
    redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
    
    mock_score = 0.45 # Simüle edilmiş sonuç
    asset = "btc"
    
    redis_key = f"precomputed_sentiment:{asset}usdt"
    payload = {
        "nlp_score": mock_score,
        "updated_at": datetime.now(timezone.utc).timestamp(),
        "source": "azure_function"
    }
    
    # Skor 10 dakika geçerli
    await redis_client.setex(redis_key, 600, json.dumps(payload))
    await redis_client.aclose()
    logging.info(f"[{asset.upper()}] NLP skoru Redis'e basıldı: {mock_score}")

@app.timer_trigger(schedule="*/5 * * * *", arg_name="myTimer", run_on_startup=False, use_monitor=False)
def news_ingestion_timer(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.warning('Timer is running late!')

    logging.info('News Ingestion Timer Triggered.')
    
    # Azure Function senkron bir wrapper içinde asenkron kodumuzu çalıştırır
    asyncio.run(process_rss_and_score())
    
    logging.info('News Ingestion cycle completed successfully.')
# Builder Stage
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Pip işlemlerinden önce git ve build-essential kurmak ZORUNDASIN
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Aşama 2: Runtime (Üretim Katmanı) ────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

# Güvenlik: Root olmayan kullanıcı kullanımı
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Sadece yüklü paketleri kopyala (İmaj boyutunu düşürür)
COPY --from=builder /install /usr/local

# Proje dosyalarını ve modelleri kopyala
# Not: .dockerignore dosyan varsa gereksiz dosyalar (venv, __pycache__) elenir.
COPY agents/ ./agents/
COPY models/ ./models/

# Yetkilendirme
RUN chown -R appuser:appgroup /app

# Çevresel Değişkenler
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HTTP_PORT=8080

USER appuser

# Health Check: Ajanın yaşadığını doğrula
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen(f'http://localhost:{os.getenv(\"HTTP_PORT\", \"8080\")}/healthz')"

EXPOSE 8080

# Varsayılan olarak Master AI'ı başlat (Diğer ajanlar compose/yaml ile ezilir)
CMD ["python", "-u", "agents/master_decision_ai.py"]
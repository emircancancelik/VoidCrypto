# ── Aşama 1: Builder ─────────────────────────────────────────────────────────
# Tek bir Python sürümü tanımlıyoruz ki aşağıda çatışma olmasın
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

# Gereksiz önbellekleri kapat, performansı artır
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# C derleyicileri (Numpy/Pandas için gerekli olabilir ama Git'e artık gerek yok)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# DİNAMİK BAĞIMLILIK SEÇİMİ (Burası mimarinin kalbi)
# Compose dosyasından hangi ajanı inşa ediyorsak onun .txt'si gelecek
ARG REQ_FILE=base.txt

# Önce base.txt, sonra hedef ajanın gereksinim dosyası
COPY requirements/base.txt requirements/
COPY requirements/${REQ_FILE} requirements/

# Seçilen gereksinimleri /install dizinine kur
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements/${REQ_FILE}

# ── Aşama 2: Runtime (Üretim Katmanı) ────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# Güvenlik: Root olmayan kullanıcı
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Sadece derlenmiş paketleri builder'dan kopyala
COPY --from=builder /install /usr/local

# Proje kodlarını kopyala
COPY agents/ ./agents/

RUN chown -R appuser:appgroup /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

# Hangi ajanın başlatılacağını çevresel değişkenle alıyoruz
ARG AGENT_SCRIPT=master_decision_ai.py
ENV RUN_SCRIPT=agents/${AGENT_SCRIPT}

# Çalıştır
CMD ["sh", "-c", "python -u ${RUN_SCRIPT}"]
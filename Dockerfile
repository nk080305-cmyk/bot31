FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-heb \
        tesseract-ocr-eng \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Playwright browsers ────────────────────────────────────────────────────────
RUN playwright install --with-deps chromium

# ── Application code ───────────────────────────────────────────────────────────
COPY . .

# ── Tmp directory ─────────────────────────────────────────────────────────────
RUN mkdir -p /tmp/bot31

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "bot.main"]

FROM python:3.11-slim

# Install system dependencies: Tesseract (heb+eng), OpenCV runtime libs, poppler
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-heb \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot/ ./bot/

# Runtime data directories are provided by Docker volume mounted at /data
RUN mkdir -p /data/cases

CMD ["python", "-m", "bot.main"]

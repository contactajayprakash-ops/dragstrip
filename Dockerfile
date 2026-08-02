FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# writable dirs for the repo cache and saved races (Spaces runs as uid 1000)
RUN mkdir -p /data/repos /data/races && chmod -R 777 /data
ENV DRAGSTRIP_CACHE=/data/repos \
    DRAGSTRIP_RACES=/data/races \
    PORT=7860

EXPOSE 7860
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]

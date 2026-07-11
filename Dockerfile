FROM --platform=linux/amd64 python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py audio_transcriber.py pipeline.py schemas.py video_processor.py run_headless.py entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

ENTRYPOINT ["/app/entrypoint.sh"]

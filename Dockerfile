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
# Baked in at build time because the AMD hackathon evaluation harness does
# not support injecting secrets into submitted containers and the submission
# cannot be edited to add one. Supplied via --build-arg in CI (see
# .github/workflows/docker-publish.yml), never committed in plaintext.
ARG FIREWORKS_API_KEY
ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY}

ENTRYPOINT ["/app/entrypoint.sh"]

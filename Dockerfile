
# Stage 1: Install dependencies and pre-download models
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y pkg-config && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Install requirements first (cached unless requirements.txt changes)
COPY /app/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Install model dependencies and download models (cached unless scripts or model logic changes)
# COPY /scripts/install_bge-m3.py ./install_bge-m3.py
# RUN pip install huggingface_hub \
#  && python -c "from faster_whisper import WhisperModel; WhisperModel('turbo')" \
#  && python install_bge-m3.py

# Stage 2: Runtime image with only what's needed
FROM python:3.10-slim AS fastapi

RUN apt-get update && apt-get install -y pkg-config ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Add source code (this will be the only part rebuilt regularly)
COPY ./app /code/app
RUN pip install urllib3 yt_dlp
EXPOSE 5000
CMD ["python", "./app/app.py"]


FROM python:3.10-slim AS milvus

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Download and setup Milvus
RUN curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
RUN chmod +x standalone_embed.sh

# Start Milvus (you may need to modify this to run in foreground)
CMD ["bash", "standalone_embed.sh", "start"]

FROM node:lts-alpine AS fe

WORKDIR /app

COPY /fe/package*.json ./

RUN npm install

COPY /fe/ .

RUN npm run build

EXPOSE 3000
CMD [ "npm", "run", "serve" ]


FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install build tools, C compiler, ffmpeg
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
  gcc build-essential cmake ffmpeg libclang-dev \
  && rm -rf /var/lib/apt/lists/*

# Set env vars for safe cache locations
ENV CC=gcc \
    TRITON_CACHE_DIR=/tmp/triton \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/hub

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces expects the app on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

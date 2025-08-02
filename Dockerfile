FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CC=gcc \
    TRITON_CACHE_DIR=/tmp/triton \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/hub

# Install build tools and clean up
RUN apt-get update && apt-get install -y \
  gcc build-essential cmake ffmpeg libclang-dev tzdata \
  && rm -rf /var/lib/apt/lists/*

# ✅ Add dummy user with UID 1000 (to fix getpass.getuser())
RUN useradd -m -u 1000 appuser
USER appuser

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
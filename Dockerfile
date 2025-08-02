FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV TRITON_CACHE_DIR=/tmp/triton
ENV HF_HOME=/tmp/hf
ENV TRANSFORMERS_CACHE=/tmp/hf/hub

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

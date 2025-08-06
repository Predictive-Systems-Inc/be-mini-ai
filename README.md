---
title: Be Mini Ai
emoji: 👀
colorFrom: yellow
colorTo: yellow
sdk: docker
pinned: false
license: apache-2.0
short_description: better-ed mini
---

To setup runpod server (5090):
1. Follow unsloth tutorial, https://docs.unsloth.ai/basics/training-llms-with-blackwell-rtx-50-series-and-unsloth
2. Update transformer to "transformers>=4.53.0"
3. Install kernel on jupyter notebook:
  pip install ipywidgets==8.1.1
  pip install ipykernel ipython
  python -m ipykernel install --user --name=uv_env --display-name "Unsloth (uv)"
4. You can now restart kernel and choose this new environment

Connect to runpod - 
ssh ymhhvwsjp7v90d-64410b91@ssh.runpod.io -i ~/.ssh/id_ed25519

uv pip install timm librosa
cd workspace/unsloth-blackwell
source .venv/bin/activate

apt install vim 

cd uvicorn

uvicorn be-main:app --host 0.0.0.0 --port 8080

To kill runpod uvicorn:
lsof -ti:8080 | xargs kill -9
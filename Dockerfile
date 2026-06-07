# CPU runtime for the Vaani speech-to-speech cascade.
# Models are NOT baked in (keeps the image lean + license-clean); they download
# to the HF cache on first run. Mount a volume at /root/.cache/huggingface to
# persist them. Default CMD runs the zero-download dummy pipeline so the image
# is demonstrable immediately; override CMD for the real cascade.
FROM python:3.12-slim

# libsndfile for soundfile. ffmpeg is intentionally omitted: torch is pinned
# <2.9 so coqui-tts uses torchaudio IO rather than torchcodec.
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch first, from the PyTorch CPU index (avoids the default CUDA wheels).
RUN pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cpu

# Heavy stack next (own layer for caching), then the package itself.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY data/samples/manifest.example.json ./data/samples/manifest.example.json
RUN pip install --no-cache-dir --no-deps -e .

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    COQUI_TOS_AGREED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Demonstrable with no downloads. For the real pipeline (downloads models):
#   docker run -v hf-cache:/root/.cache/huggingface vaani \
#       python scripts/run_eval.py
CMD ["python", "scripts/run_eval.py", "--config", "configs/dummy.yaml"]

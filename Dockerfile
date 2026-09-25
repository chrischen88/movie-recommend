# One image for the whole app: FastAPI serves the API and the built UI on :8080.
# Build: fly deploy (or docker build .). See docs/DEPLOY.md.

# ---- UI
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- API
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    DATA_DIR=/data \
    ANONYMIZED_TELEMETRY=False
WORKDIR /app/backend

# Dependencies from pyproject.toml, in their own layer so code changes don't reinstall them.
COPY backend/pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt

COPY backend/app ./app

# Bake the embedding model's ONNX export in, so a fresh machine never downloads
# it. Keep this in sync with EMBEDDING_MODEL (config.py); changing the model
# means rebuilding.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from app.embeddings import OnnxEmbedder; OnnxEmbedder.download('${EMBEDDING_MODEL}')"
ENV HF_HUB_OFFLINE=1
COPY deploy/start.sh /usr/local/bin/start
COPY --from=web /web/dist /app/frontend/dist

EXPOSE 8080
CMD ["start"]

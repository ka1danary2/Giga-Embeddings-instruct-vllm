# Pin a GPU OpenAI-compatible image that still speaks the pooling APIs we shim.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.10.2
FROM ${VLLM_IMAGE}

USER root
WORKDIR /opt/vllm_gigarembed

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir "bitsandbytes>=0.43.0" "einops>=0.7.0" "accelerate>=0.33.0"

# Load only this general plugin (optional; omit to load all installed plugins).
ENV VLLM_PLUGINS=register_gigarembed
ENV GIGA_LOAD_IN_4BIT=1
ENV GIGA_HF_MAX_BATCH=8

# Default CMD from base image is still `vllm serve ...` (compose/command overrides).

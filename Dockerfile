# Pin a GPU OpenAI-compatible image that still speaks the pooling APIs we shim.
ARG VLLM_IMAGE=vllm/vllm-openai:v0.10.2
FROM ${VLLM_IMAGE}

USER root
WORKDIR /opt/vllm_gigarembed

COPY pyproject.toml ./
COPY src ./src
RUN if [ ! -f README.md ]; then printf '%s\n' '# vllm-gigarembed' > README.md; fi

# --no-deps: never upgrade transformers/torch that ship with the vLLM image.
RUN pip install --no-cache-dir --no-deps -e . \
    && pip install --no-cache-dir --no-deps "einops>=0.7.0" \
    && pip install --no-cache-dir "bitsandbytes>=0.43.0" "accelerate>=0.33.0"

ENV VLLM_PLUGINS=register_gigarembed
ENV GIGA_LOAD_IN_4BIT=1
ENV GIGA_HF_MAX_BATCH=8
# GigarEmbedConfig imports CONFIG_MAPPING; vLLM spawn cannot unpickle it.
ENV VLLM_ENABLE_V1_MULTIPROCESSING=0
ENV VLLM_WORKER_MULTIPROC_METHOD=fork

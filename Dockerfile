ARG VLLM_IMAGE=vllm/vllm-openai:v0.10.2
FROM ${VLLM_IMAGE}

USER root
WORKDIR /opt/vllm_gigarembed

COPY pyproject.toml requirements-docker.txt ./
COPY src ./src
RUN if [ ! -f README.md ]; then printf '%s\n' '# vllm-gigarembed' > README.md; fi

RUN pip install --no-cache-dir -r requirements-docker.txt \
    && pip install --no-cache-dir --no-deps -e .

ENV VLLM_PLUGINS=register_gigarembed
ENV GIGA_LOAD_IN_4BIT=1
ENV GIGA_HF_MAX_BATCH=8
ENV VLLM_ENABLE_V1_MULTIPROCESSING=0
ENV VLLM_WORKER_MULTIPROC_METHOD=fork

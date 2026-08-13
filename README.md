# Giga-Embeddings-instruct for vLLM

Out-of-tree **vLLM plugin** that registers HuggingFace architecture `GigarEmbedModel`
so `vllm serve … --runner pooling` can expose OpenAI-compatible `/v1/embeddings`.

> **MVP design:** vLLM owns HTTP/scheduling; the forward path calls the official
> `transformers` + `trust_remote_code` GigarEmbed implementation (latent-attention
> pooling). This is **not** a native vLLM MLA port — embeddings match HF, throughput
> is closer to a HF server than to optimized vLLM kernels.

## Install (into a vLLM environment)

```bash
pip install -e /path/to/Giga-Embeddings-instruct-vllm
# bitsandbytes needed for NF4 checkpoints
pip install bitsandbytes einops accelerate
```

Entry point: `vllm.general_plugins` → `register_gigarembed`.

## Serve

```bash
export VLLM_PLUGINS=register_gigarembed   # optional filter
export GIGA_LOAD_IN_4BIT=1                # auto for paths containing nf4/4bit

vllm serve /path/to/Giga-Embeddings-instruct-4bit-nf4 \
  --runner pooling \
  --trust-remote-code \
  --served-model-name Giga-Embeddings-instruct \
  --host 0.0.0.0 \
  --port 8090 \
  --max-model-len 4096
```

Smoke:

```bash
curl -s http://127.0.0.1:8090/v1/models | jq .
curl -s http://127.0.0.1:8090/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Giga-Embeddings-instruct","input":"тест"}' | jq '.data[0].embedding | length'
# expect 2048
```

Instruct prefixes for queries still belong in the **client** (RAGFlow `GIGA_TASK`),
same as the HF README: `Instruct: …\nQuery: …`.

## Docker

```bash
export GIGA_MODEL_HOST_PATH=/abs/path/to/Giga-Embeddings-instruct-4bit-nf4
docker compose -f docker-compose.example.yml up -d --build
```

Or:

```bash
docker build -t vllm-gigarembed:local .
docker run --gpus all --rm -p 8090:8090 \
  -v /abs/path/to/model:/models/giga:ro \
  vllm-gigarembed:local \
  --model /models/giga --runner pooling --trust-remote-code \
  --served-model-name Giga-Embeddings-instruct \
  --host 0.0.0.0 --port 8090 --max-model-len 4096
```

Default base image pin: `vllm/vllm-openai:v0.10.2` (override with build-arg `VLLM_IMAGE`).
The pooler code shims 0.10 (`build_output`) and newer list-style poolers.

## Env

| Variable | Default | Meaning |
|----------|---------|---------|
| `VLLM_PLUGINS` | (all) | Set to `register_gigarembed` to load only this plugin |
| `GIGA_LOAD_IN_4BIT` | auto | `1`/`0` force NF4 bitsandbytes load |
| `GIGA_HF_MAX_BATCH` | `8` | HF forward batch size inside the pooler |

## Layout

```
src/vllm_gigarembed/
  __init__.py   # register()
  model.py      # GigarEmbedForPooling
  pooler.py     # GigarHFPooler (calls HF return_embeddings=True)
  loading.py    # AutoModel + BitsAndBytesConfig
```

## RAGFlow

Point the existing `GigaSberEmbed` / compose service at this image instead of stock
`vllm/vllm-openai:latest`, keep `http://giga-embeddings:8090` and model name
`Giga-Embeddings-instruct`.

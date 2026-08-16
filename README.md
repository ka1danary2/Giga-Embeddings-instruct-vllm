# vllm-gigarembed

Плагин vLLM для `Giga-Embeddings-instruct` (архитектура `GigarEmbedModel`, NF4).

Веса скачиваются отдельно, в git не кладём (`models/`, `cache/` в `.gitignore`).

Зависимости: `requirements.txt` (полный стек, pin vllm 0.10.2). В Docker базовый образ уже содержит vllm — ставится только `requirements-docker.txt`.

Сборка и запуск:

```bash
export GIGA_MODEL_HOST_PATH=/path/to/Giga-Embeddings-instruct-4bit-nf4

docker build -t vllm-gigarembed:local .
docker compose -f docker-compose.example.yml up -d
```

Проверка:

```bash
curl http://127.0.0.1:8090/v1/models
curl -sS -X POST http://127.0.0.1:8090/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"Giga-Embeddings-instruct","input":"тестовый текст"}'
```

В RAGFlow: `GIGA_VLLM_PLUGIN_PATH` → этот репозиторий, провайдер Sber-Giga → `http://giga-embeddings:8090`.

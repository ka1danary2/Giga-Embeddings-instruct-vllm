# vllm-gigarembed

Плагин vLLM для `Giga-Embeddings-instruct` (архитектура `GigarEmbedModel`, NF4).

Сборка и запуск:

```bash
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


from __future__ import annotations

import logging
import os
from typing import Any

import torch
from transformers import AutoModel, BitsAndBytesConfig

logger = logging.getLogger("vllm_gigarembed")


def _want_4bit(model_path: str) -> bool:
    env = os.getenv("GIGA_LOAD_IN_4BIT", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    lowered = model_path.lower()
    return "nf4" in lowered or "4bit" in lowered or "4-bit" in lowered


def load_gigar_embed(model_path: str, device: str | torch.device) -> Any:
    device = torch.device(device)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if _want_4bit(model_path):
        logger.info("Loading GigarEmbed with bitsandbytes NF4 from %s", model_path)
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": str(device)}
    else:
        logger.info("Loading GigarEmbed (bf16) from %s", model_path)
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModel.from_pretrained(model_path, **kwargs)
    if "device_map" not in kwargs:
        model = model.to(device)
    model.eval()
    return model

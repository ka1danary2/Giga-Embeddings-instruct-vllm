"""vLLM config hooks for GigarEmbedModel."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger("vllm_gigarembed")


class GigarEmbedModelConfig:
    """Disable vLLM's bitsandbytes loader for NF4 GigarEmbed checkpoints."""

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        model_config = vllm_config.model_config
        load_config = vllm_config.load_config

        # vLLM auto-detects NF4 in config.json and forces load_format=bitsandbytes
        # even when --load-format dummy is passed. GigarEmbedModel is not supported
        # by vLLM's BitsAndBytesModelLoader; the plugin loads HF weights instead.
        if model_config.quantization == "bitsandbytes":
            logger.info(
                "GigarEmbed: disabling vLLM bitsandbytes path; "
                "NF4 weights are loaded via HuggingFace in the plugin."
            )
            model_config.quantization = None
            load_config.load_format = "dummy"

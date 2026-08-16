from __future__ import annotations


def register() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from vllm_gigarembed.config import GigarEmbedModelConfig

    arch = "GigarEmbedModel"
    target = "vllm_gigarembed.model:GigarEmbedForPooling"
    if arch not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(arch, target)
    MODELS_CONFIG_MAP.setdefault(arch, GigarEmbedModelConfig)

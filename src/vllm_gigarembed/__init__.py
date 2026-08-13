"""vLLM out-of-tree plugin for GigarEmbedModel (Giga-Embeddings-instruct)."""

from __future__ import annotations

__version__ = "0.1.0"


def register() -> None:
    """Entry point for ``vllm.general_plugins``."""
    from vllm import ModelRegistry

    arch = "GigarEmbedModel"
    target = "vllm_gigarembed.model:GigarEmbedForPooling"
    supported = ModelRegistry.get_supported_archs()
    if arch not in supported:
        ModelRegistry.register_model(arch, target)

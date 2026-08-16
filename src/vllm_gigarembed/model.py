from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from vllm_gigarembed.loading import load_gigar_embed
from vllm_gigarembed.pooler import GigarHFPooler

logger = logging.getLogger("vllm_gigarembed")


def _default_pooling_decorator():
    try:
        from vllm.model_executor.models.interfaces_base import default_pooling_type

        try:
            return default_pooling_type(seq_pooling_type="MEAN")
        except TypeError:
            return default_pooling_type("MEAN")
    except Exception:

        def identity(cls):
            return cls

        return identity


@_default_pooling_decorator()
class GigarEmbedForPooling(nn.Module):
    is_pooling_model = True

    packed_modules_mapping = {"_plugin": ["_plugin"]}

    def __init__(
        self,
        vllm_config: Any = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del prefix
        if vllm_config is None:
            vllm_config = kwargs.pop("vllm_config", None)
        if vllm_config is None:
            raise TypeError("vllm_config is required")

        model_config = vllm_config.model_config
        device = torch.device(vllm_config.device_config.device)
        model_path = model_config.model
        self.hidden_size = int(
            getattr(model_config.hf_config, "hidden_size", None)
            or getattr(getattr(model_config.hf_config, "text_config", None), "hidden_size", None)
            or 2048
        )

        hf_model = load_gigar_embed(model_path, device=device)
        object.__setattr__(self, "hf_model", hf_model)

        max_batch = int(os.getenv("GIGA_HF_MAX_BATCH", "8"))
        embed_pooler = GigarHFPooler(hf_model, device=device, max_batch=max_batch)
        self.pooler = self._build_dispatch_pooler(embed_pooler)
        logger.info(
            "GigarEmbedForPooling ready path=%s hidden=%s device=%s",
            model_path,
            self.hidden_size,
            device,
        )

    @staticmethod
    def _build_dispatch_pooler(embed_pooler: GigarHFPooler) -> Any:
        from vllm.model_executor.layers.pooler import DispatchPooler

        return DispatchPooler({"embed": embed_pooler, "encode": embed_pooler})

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat = input_ids.reshape(-1)
        return torch.zeros(
            (flat.shape[0], self.hidden_size),
            dtype=torch.float32,
            device=flat.device,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions, intermediate_tensors, inputs_embeds
        if input_ids is None:
            raise ValueError("GigarEmbedForPooling requires input_ids")
        if input_ids.dim() == 1:
            num_tokens = input_ids.shape[0]
        elif input_ids.dim() == 2:
            num_tokens = input_ids.shape[0] * input_ids.shape[1]
        else:
            num_tokens = input_ids.numel()
        return torch.zeros(
            (num_tokens, self.hidden_size),
            dtype=torch.float32,
            device=input_ids.device,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        for _ in weights:
            pass
        return set()

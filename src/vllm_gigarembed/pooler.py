"""Pooling head that runs the real GigarEmbed HF forward (latent-attention)."""

from __future__ import annotations

import logging
from collections.abc import Set
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from vllm.v1.pool.metadata import PoolingMetadata

logger = logging.getLogger("vllm_gigarembed")


def _get_prompt_token_ids(pooling_metadata: "PoolingMetadata") -> list[torch.Tensor]:
    if hasattr(pooling_metadata, "get_prompt_token_ids_cpu"):
        return pooling_metadata.get_prompt_token_ids_cpu()
    if hasattr(pooling_metadata, "get_prompt_token_ids"):
        return pooling_metadata.get_prompt_token_ids()
    try:
        from vllm.model_executor.layers.pooler import get_prompt_token_ids

        return get_prompt_token_ids(pooling_metadata)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Cannot read prompt_token_ids from pooling_metadata; "
            "set requires_token_ids=True in get_pooling_updates."
        ) from exc


def _wrap_pooler_output(embeddings: list[torch.Tensor]) -> Any:
    """Compatibility shim for vLLM 0.10 (PoolerOutput) vs newer (raw list)."""
    try:
        import inspect

        from vllm.model_executor.layers.pooler import DispatchPooler, build_output

        src = inspect.getsource(DispatchPooler.forward)
        # v0.10 aggregates ``group_output.outputs``; newer extends a raw list.
        if "group_output.outputs" in src or "PoolingSequenceGroupOutput" in src:
            return build_output(embeddings)
    except Exception:  # noqa: BLE001
        pass
    return embeddings


class GigarHFPooler(nn.Module):
    """Ignore vLLM hidden states; embed with HuggingFace GigarEmbedModel."""

    def __init__(self, hf_model: nn.Module, device: torch.device, max_batch: int = 8):
        super().__init__()
        object.__setattr__(self, "hf_model", hf_model)
        self.device = device
        self.max_batch = max(1, int(max_batch))

    def get_supported_tasks(self) -> Set[str]:
        return {"embed", "encode"}

    def get_pooling_updates(self, task: str) -> Any:
        from vllm.model_executor.layers.pooler import PoolingParamsUpdate

        return PoolingParamsUpdate(requires_token_ids=True)

    @torch.inference_mode()
    def _embed_batch(self, sequences: list[torch.Tensor]) -> list[torch.Tensor]:
        if not sequences:
            return []

        lengths = [int(s.numel()) for s in sequences]
        max_len = max(lengths)
        batch = torch.zeros(
            (len(sequences), max_len),
            dtype=torch.long,
            device=self.device,
        )
        attn = torch.zeros(
            (len(sequences), max_len),
            dtype=torch.long,
            device=self.device,
        )
        for i, (seq, length) in enumerate(zip(sequences, lengths)):
            batch[i, :length] = seq.to(device=self.device, dtype=torch.long)
            attn[i, :length] = 1

        out = self.hf_model(
            input_ids=batch,
            attention_mask=attn,
            return_embeddings=True,
        )
        if isinstance(out, tuple):
            out = out[0]
        if not torch.is_tensor(out):
            raise TypeError(f"Unexpected GigarEmbed output type: {type(out)}")
        # Already L2-normalized inside modeling_gigarembed.mean_pool
        return [out[i].float() for i in range(out.shape[0])]

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: "PoolingMetadata",
    ) -> Any:
        del hidden_states  # Gigar latent-attention pooling is done inside HF.
        token_ids = _get_prompt_token_ids(pooling_metadata)
        embeddings: list[torch.Tensor] = []
        for start in range(0, len(token_ids), self.max_batch):
            chunk = token_ids[start : start + self.max_batch]
            embeddings.extend(self._embed_batch(chunk))
        return _wrap_pooler_output(embeddings)

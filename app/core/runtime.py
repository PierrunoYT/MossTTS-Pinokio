"""Runtime resolution: device, dtype, and attention backend selection.

A :class:`RuntimeConfig` bundles the immutable per-process choices (device +
attention implementation) that every tab passes around, replacing the loose
``(device_str, attn_implementation)`` argument pairs.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Optional

import torch


def pick_dtype(device: torch.device) -> torch.dtype:
    """Use native bf16 where supported, fp16 on older CUDA cards, fp32 on CPU."""
    if device.type == "cuda":
        with torch.cuda.device(device):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def resolve_attn_implementation(
    requested: str, device: torch.device, dtype: torch.dtype
) -> Optional[str]:
    """Pick the best attention backend for the given device and dtype.

    Returns ``None`` to mean "let transformers decide" only when the caller
    explicitly asked for ``none``; otherwise a concrete backend string.
    """
    requested_norm = (requested or "").strip().lower()

    if requested_norm == "none":
        return None

    if requested_norm not in {"", "auto"}:
        return requested

    # Prefer FlashAttention 2 when installed and the GPU/dtype support it.
    if (
        device.type == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return "flash_attention_2"

    if device.type == "cuda":
        return "sdpa"

    return "eager"


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable runtime choices shared across tabs."""

    device_str: str
    attn_implementation: str

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_str if torch.cuda.is_available() else "cpu")

    @property
    def dtype(self) -> torch.dtype:
        return pick_dtype(self.device)

    def resolved_attn(self) -> Optional[str]:
        return resolve_attn_implementation(self.attn_implementation, self.device, self.dtype)

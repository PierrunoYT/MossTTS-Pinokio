"""GPU residency manager.

A size-bounded LRU keeps at most ``max_resident_models()`` heavy checkpoints on
the GPU at once and *actively frees* evicted ones back to the CUDA allocator.
This is what stops a 24 GB card from overflowing (and the NVIDIA driver from
spilling weights to system RAM, which makes generation glacially slow) when the
user hops between the 8B TTS/TTSD/SoundEffect tabs.
"""

from __future__ import annotations

import gc
import os
from collections import OrderedDict
from typing import Optional

import torch

from config import DEFAULT_MODEL_CACHE_SIZE, MODEL_CACHE_SIZE_ENV_VAR

# cache_key -> payload tuple (model/inferencer, ...)
_RESIDENTS: "OrderedDict[tuple, tuple]" = OrderedDict()


def max_resident_models() -> int:
    raw = os.getenv(MODEL_CACHE_SIZE_ENV_VAR)
    if raw is None:
        return DEFAULT_MODEL_CACHE_SIZE
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MODEL_CACHE_SIZE


def release_gpu_memory() -> None:
    """Return freed allocations to the OS so the next model gets a clean block."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def free_after_generation(device: torch.device) -> None:
    """Light cleanup after a single generation.

    Drops cached intermediate allocations so a long SFX/dialogue run doesn't
    leave the KV cache and activation buffers pinned until the next eviction.
    Cheaper than :func:`release_gpu_memory` (no ``ipc_collect``).
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()


def evict_to_make_room(incoming: int = 1) -> None:
    """Drop oldest residents until ``incoming`` more models fit, then free VRAM."""
    budget = max_resident_models()
    freed = False
    while _RESIDENTS and len(_RESIDENTS) + incoming > budget:
        _key, payload = _RESIDENTS.popitem(last=False)
        del payload
        freed = True
    if freed:
        release_gpu_memory()


def unload_all_models() -> None:
    """Evict every cached model and free its GPU memory."""
    _RESIDENTS.clear()
    release_gpu_memory()


def cache_get(cache_key: tuple) -> Optional[tuple]:
    cached = _RESIDENTS.get(cache_key)
    if cached is not None:
        _RESIDENTS.move_to_end(cache_key)  # mark most-recently-used
    return cached


def cache_put(cache_key: tuple, payload: tuple) -> None:
    _RESIDENTS[cache_key] = payload
    _RESIDENTS.move_to_end(cache_key)

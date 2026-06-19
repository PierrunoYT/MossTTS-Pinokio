"""Weight quantization resolution (bitsandbytes, CUDA only).

``auto`` (the Pinokio default) picks 4-bit NF4 on modest GPUs so the 8B
checkpoints fit in ~6 GB instead of ~16 GB and never trigger the NVIDIA
sysmem-fallback slowdown. It degrades gracefully to bf16 when bitsandbytes is
missing or the card has plenty of VRAM.
"""

from __future__ import annotations

import os

import torch

from config import AUTO_QUANT_VRAM_THRESHOLD_GB, QUANTIZATION_ENV_VAR

_NONE_ALIASES = {"", "none", "off", "no", "false", "0"}
_4BIT_ALIASES = {"4bit", "int4", "nf4"}
_8BIT_ALIASES = {"8bit", "int8"}


def bitsandbytes_available() -> bool:
    try:
        from transformers import BitsAndBytesConfig  # noqa: F401
        import bitsandbytes  # noqa: F401
        # bitsandbytes loads weights via a ``device_map``, which transformers
        # only honours when ``accelerate`` is installed. Without it, 4-bit/8-bit
        # loading raises at ``from_pretrained``; treat it as unavailable so we
        # degrade to bf16 instead of crashing mid-generation.
        import accelerate  # noqa: F401
        return True
    except Exception:
        return False


def _auto_quantization() -> str:
    """Pick 4-bit on modest CUDA GPUs when bitsandbytes is available, else bf16."""
    if not torch.cuda.is_available():
        return "none"
    if not bitsandbytes_available():
        print(
            "  [quantization] auto: bitsandbytes not installed — running in bf16. "
            "Install it (`pip install bitsandbytes`) to enable 4-bit."
        )
        return "none"
    try:
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return "none"
    if total_gb <= AUTO_QUANT_VRAM_THRESHOLD_GB:
        print(f"  [quantization] auto: {total_gb:.0f} GB GPU → using 4-bit (nf4).")
        return "4bit"
    print(f"  [quantization] auto: {total_gb:.0f} GB GPU → bf16 (no quantization).")
    return "none"


def resolve_quantization() -> str:
    """Resolve the configured quantization mode to a concrete one.

    Returns one of ``"none"``, ``"4bit"``, ``"8bit"`` (other aliases collapse to
    these). ``auto`` is resolved against the live GPU.
    """
    mode = (os.getenv(QUANTIZATION_ENV_VAR) or "none").strip().lower()
    if mode == "auto":
        return _auto_quantization()
    if mode in _NONE_ALIASES:
        return "none"
    if mode in _4BIT_ALIASES:
        return "4bit"
    if mode in _8BIT_ALIASES:
        return "8bit"
    return mode  # unknown → let build_quantization_config raise a clear error


def build_quantization_config(mode: str, compute_dtype: torch.dtype):
    """Return a ``BitsAndBytesConfig`` for ``mode``, or ``None`` for bf16.

    Raises a clear error if quantization is requested but unsupported (no CUDA
    or bitsandbytes missing) so the user isn't left guessing.
    """
    if mode in _NONE_ALIASES:
        return None

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{QUANTIZATION_ENV_VAR}={mode!r} requires a CUDA GPU; "
            "quantization is not available on CPU."
        )

    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401  (import to verify it's installed)
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            f"{QUANTIZATION_ENV_VAR}={mode!r} requires the 'bitsandbytes' package. "
            "Install it with `pip install bitsandbytes` (CUDA only)."
        ) from exc

    if mode in _4BIT_ALIASES:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    if mode in _8BIT_ALIASES:
        return BitsAndBytesConfig(load_in_8bit=True)

    raise ValueError(
        f"Unknown {QUANTIZATION_ENV_VAR} value {mode!r}; expected 'auto', 'none', '8bit', or '4bit'."
    )

"""Core engine for the MOSS-TTS app.

This package centralises everything that used to live in the monolithic
``model_loader.py``: device/dtype resolution, GPU residency management,
quantization, downloads, audio helpers, and the shared generate→decode path.

Tabs should depend on this package rather than re-implementing model loading
or the (previously duplicated) generation/decoding logic.
"""

from core.audio import audio_to_int16, load_audio, resample_wav, truncate_reference_audio
from core.download import (
    download_model_files,
    download_model_files_for_keys,
    resolve_hf_path,
)
from core.engine import (
    Sampling,
    generate_and_decode,
    load_model,
    load_realtime_model,
)
from core.memory import unload_all_models
from core.runtime import RuntimeConfig, pick_dtype, resolve_attn_implementation

__all__ = [
    "audio_to_int16",
    "load_audio",
    "resample_wav",
    "truncate_reference_audio",
    "download_model_files",
    "download_model_files_for_keys",
    "resolve_hf_path",
    "Sampling",
    "generate_and_decode",
    "load_model",
    "load_realtime_model",
    "unload_all_models",
    "RuntimeConfig",
    "pick_dtype",
    "resolve_attn_implementation",
]

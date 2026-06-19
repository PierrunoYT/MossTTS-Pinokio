"""Backward-compatibility shim.

The model-loading logic now lives in the :mod:`core` package. This module
re-exports the public names so any external scripts (or muscle memory) that
still ``import model_loader`` keep working. New code should import from
``core`` directly.
"""

from core.audio import audio_to_int16, load_audio, resample_wav
from core.audio import truncate_reference_audio as _truncate_reference_audio
from core.download import (
    download_model_files,
    download_model_files_for_keys,
    resolve_hf_path as _resolve_hf_path,
)
from core.engine import Sampling, generate_and_decode, load_model, load_realtime_model
from core.memory import unload_all_models
from core.runtime import resolve_attn_implementation

__all__ = [
    "audio_to_int16",
    "load_audio",
    "resample_wav",
    "_truncate_reference_audio",
    "download_model_files",
    "download_model_files_for_keys",
    "_resolve_hf_path",
    "Sampling",
    "generate_and_decode",
    "load_model",
    "load_realtime_model",
    "unload_all_models",
    "resolve_attn_implementation",
]

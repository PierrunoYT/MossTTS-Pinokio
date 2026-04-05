"""Model loading, caching, and audio pre-processing utilities."""

import functools
import importlib.util
import os
import sys
import tempfile
import time
from typing import Optional

import torch
from transformers import AutoModel, AutoProcessor

from config import CODEC_MODEL_PATH, MAX_REFERENCE_DURATION_SEC, MODELS


# ---------------------------------------------------------------------------
# Attention implementation resolver
# ---------------------------------------------------------------------------

def resolve_attn_implementation(
    requested: str, device: torch.device, dtype: torch.dtype
) -> Optional[str]:
    """Pick the best attention backend for the given device and dtype."""
    requested_norm = (requested or "").strip().lower()

    if requested_norm == "none":
        return None

    if requested_norm not in {"", "auto"}:
        return requested

    # Prefer FlashAttention 2 when the package is installed and conditions are met
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


# ---------------------------------------------------------------------------
# HuggingFace path resolver (Windows workaround)
# ---------------------------------------------------------------------------

def _resolve_hf_path(repo_id: str) -> str:
    """Resolve a HuggingFace repo ID to a local snapshot path on Windows.

    Custom processor code often calls ``Path(repo_id)`` which on Windows turns
    the ``/`` in ``Org/Model`` into backslashes, producing an invalid repo ID.
    Pre-downloading with ``snapshot_download`` gives us a real local path.

    Retries up to 5 times on network errors; each attempt resumes thanks to
    HuggingFace Hub's local cache.  After each attempt we verify the snapshot
    contains at least a ``config.json`` — if not, the partial cache is purged
    so the next attempt starts fresh.
    """
    if sys.platform == "win32" and "/" in repo_id and not os.path.isdir(repo_id):
        from huggingface_hub import snapshot_download
        import shutil

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                local_path = snapshot_download(repo_id)
                # Validate the snapshot is usable (not a partial download)
                if not os.path.isfile(os.path.join(local_path, "config.json")):
                    raise RuntimeError(
                        f"Incomplete download: config.json missing in {local_path}"
                    )
                return local_path
            except Exception as exc:
                # Purge partial cache so the next attempt doesn't reuse it
                _purge_partial_cache(repo_id)
                if attempt == max_attempts:
                    raise
                wait = attempt * 5
                print(
                    f"⚠️  Download interrupted ({exc.__class__.__name__}: {exc}). "
                    f"Retrying in {wait}s… (attempt {attempt}/{max_attempts})"
                )
                time.sleep(wait)
    return repo_id


def _purge_partial_cache(repo_id: str) -> None:
    """Remove a partially-downloaded HuggingFace Hub cache for *repo_id*."""
    import shutil
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return
    # HF Hub stores models in folders like "models--Org--Repo"
    cache_dir_name = "models--" + repo_id.replace("/", "--")
    cache_path = os.path.join(HF_HUB_CACHE, cache_dir_name)
    if os.path.isdir(cache_path):
        print(f"🗑️  Purging incomplete cache: {cache_path}")
        shutil.rmtree(cache_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Reference audio truncation (prevents O(L²) OOM in the audio tokenizer)
# ---------------------------------------------------------------------------

def _truncate_reference_audio(
    audio_path: str, max_duration: float = MAX_REFERENCE_DURATION_SEC
) -> str:
    """Return ``audio_path`` unchanged if it is short enough, otherwise write a
    truncated copy to a temp file and return that path instead."""
    import librosa
    import soundfile as sf

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    max_samples = int(max_duration * sr)
    if len(y) <= max_samples:
        return audio_path

    print(
        f"⚠️  Reference audio is {len(y) / sr:.1f}s — truncating to {max_duration:.0f}s "
        f"to avoid GPU OOM in the audio tokenizer."
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y[:max_samples], sr)
    return tmp.name


# ---------------------------------------------------------------------------
# Cached model loader
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=10)
def load_model(model_key: str, device_str: str, attn_implementation: str):
    """Load and LRU-cache a model + processor pair."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model_path = MODELS[model_key]
    print(f"Loading {model_key} from {model_path}…")

    local_model_path = _resolve_hf_path(model_path)
    resolved_attn = resolve_attn_implementation(attn_implementation, device, dtype)

    processor_kwargs: dict = {"trust_remote_code": True}
    if model_key == "ttsd":
        processor_kwargs["codec_path"] = _resolve_hf_path(CODEC_MODEL_PATH)

    processor = AutoProcessor.from_pretrained(local_model_path, **processor_kwargs)

    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
        processor.audio_tokenizer.eval()

    model_kwargs: dict = {"trust_remote_code": True, "torch_dtype": dtype}
    if resolved_attn:
        model_kwargs["attn_implementation"] = resolved_attn

    model = AutoModel.from_pretrained(local_model_path, **model_kwargs).to(device)
    model.eval()

    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
    print(f"✓ {model_key} loaded")
    return model, processor, device, sample_rate

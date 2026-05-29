"""Model loading, caching, and audio pre-processing utilities."""

import functools
import importlib.util
import os
import sys
import tempfile
import time
from typing import Optional

import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer

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
    HuggingFace Hub's local cache.
    """
    if sys.platform == "win32" and "/" in repo_id and not os.path.isdir(repo_id):
        from huggingface_hub import snapshot_download

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                return snapshot_download(repo_id)
            except Exception as exc:
                if attempt == max_attempts:
                    raise
                wait = attempt * 5
                print(
                    f"⚠️  Download interrupted ({exc.__class__.__name__}: {exc}). "
                    f"Retrying in {wait}s… (attempt {attempt}/{max_attempts})"
                )
                time.sleep(wait)
    return repo_id


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

def _snapshot_download_repo(repo_id: str) -> None:
    """Download one repo snapshot into the Hugging Face cache with retries."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id}…")
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            snapshot_download(repo_id)
            print(f"✓ {repo_id} downloaded")
            return
        except Exception as exc:
            if attempt == max_attempts:
                raise
            wait = attempt * 5
            print(
                f"⚠️  Download interrupted ({exc.__class__.__name__}: {exc}). "
                f"Retrying in {wait}s… (attempt {attempt}/{max_attempts})"
            )
            time.sleep(wait)


def _repos_for_download(model_keys: list[str]) -> list[str]:
    """HF repo IDs needed for inference: main checkpoint(s) + shared audio tokenizer.

    MossTTSDelayProcessor loads the codec from ``OpenMOSS-Team/MOSS-Audio-Tokenizer``
    by default (tokenizer + weights live in that repo). The realtime stack also
    loads that codec explicitly. Deduplicate while preserving order.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for key in model_keys:
        rid = MODELS[key]
        if rid not in seen:
            seen.add(rid)
            ordered.append(rid)
    if CODEC_MODEL_PATH not in seen:
        ordered.append(CODEC_MODEL_PATH)
    return ordered


def download_model_files_for_keys(model_keys: list[str]) -> str:
    """Download all Hugging Face repos used when loading these model keys (main + codec)."""
    repos = _repos_for_download(model_keys)
    for repo_id in repos:
        _snapshot_download_repo(repo_id)
    return f"✅ Downloaded successfully: {', '.join(repos)}"


def download_model_files(model_key: str) -> str:
    """Download model files to the HuggingFace cache without loading into GPU.

    Fetches the tab's main checkpoint repo and the shared MOSS-Audio-Tokenizer
    (processor, tokenizer assets, and codec weights), matching ``load_model`` /
    ``load_realtime_model`` behavior.

    Returns a status message.
    """
    return download_model_files_for_keys([model_key])


@functools.lru_cache(maxsize=6)
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


@functools.lru_cache(maxsize=1)
def load_realtime_model(device_str: str, attn_implementation: str):
    """Load and LRU-cache the MOSS-TTS-Realtime model + codec + inferencer."""
    import sys
    import os

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model_path = MODELS["realtime"]
    codec_path = CODEC_MODEL_PATH
    print(f"Loading realtime model from {model_path}…")

    local_model_path = _resolve_hf_path(model_path)
    local_codec_path = _resolve_hf_path(codec_path)
    resolved_attn = resolve_attn_implementation(attn_implementation, device, dtype)

    # Ensure MOSS-TTS repo paths are importable
    moss_tts_root = os.path.join(os.path.dirname(__file__), "MOSS-TTS")
    moss_tts_realtime_dir = os.path.join(moss_tts_root, "moss_tts_realtime")
    for p in [moss_tts_root, moss_tts_realtime_dir]:
        if p not in sys.path and os.path.isdir(p):
            sys.path.insert(0, p)

    from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
    from inferencer import MossTTSRealtimeInference

    model_kwargs = {"torch_dtype": dtype}
    if resolved_attn:
        model_kwargs["attn_implementation"] = resolved_attn

    model = MossTTSRealtime.from_pretrained(local_model_path, **model_kwargs).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    # Load the codec in float32 regardless of dtype: the codec model has internal
    # operations that explicitly cast tensors to .float() for numerical stability,
    # and those tensors are then passed back into conv/linear layers — causing
    # dtype mismatches if the layer weights are bf16.
    codec = AutoModel.from_pretrained(
        local_codec_path, trust_remote_code=True, torch_dtype=torch.float32
    ).eval().to(device)

    inferencer = MossTTSRealtimeInference(
        model, tokenizer,
        max_length=5000,
        codec=codec,
        codec_sample_rate=24000,
        codec_encode_kwargs={"chunk_duration": 8},
    )

    print("✓ realtime model loaded")
    return inferencer, codec, device, 24000

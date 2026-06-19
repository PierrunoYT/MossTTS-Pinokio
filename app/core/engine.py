"""The model engine: loading, caching, and the single shared generate→decode path.

Every speech tab (TTS, TTSD, VoiceGenerator, SoundEffect) funnels through
:func:`generate_and_decode`, eliminating the four near-identical
``model.generate(...) + processor.decode(...) + int16`` blocks that used to live
in the individual tabs. Realtime keeps its own streaming loop but shares the
loader and residency budget.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from config import CODEC_MODEL_PATH, MODELS
from core.audio import audio_to_int16
from core.download import resolve_hf_path
from core.memory import cache_get, cache_put, evict_to_make_room, free_after_generation
from core.quant import build_quantization_config, resolve_quantization
from core.runtime import RuntimeConfig, pick_dtype, resolve_attn_implementation


@dataclass
class Sampling:
    """Audio sampling parameters shared by every MOSS ``generate`` call."""

    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0

    def as_generate_kwargs(self) -> dict:
        return {
            "audio_temperature": float(self.temperature),
            "audio_top_p": float(self.top_p),
            "audio_top_k": int(self.top_k),
            "audio_repetition_penalty": float(self.repetition_penalty),
        }


def _runtime(device_str: str, attn_implementation: str) -> RuntimeConfig:
    return RuntimeConfig(device_str=device_str, attn_implementation=attn_implementation)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_model(model_key: str, device_str: str, attn_implementation: str):
    """Load a (model, processor, device, sample_rate) tuple, keeping at most
    ``max_resident_models()`` resident in GPU memory (others freed first)."""
    quantization = resolve_quantization()
    cache_key = ("model", model_key, device_str, attn_implementation, quantization)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    # Free other residents *before* allocating so the freed VRAM is available
    # to the incoming 8B checkpoint (critical on 24 GB cards).
    evict_to_make_room(incoming=1)

    rt = _runtime(device_str, attn_implementation)
    device, dtype = rt.device, rt.dtype

    model_path = MODELS[model_key]
    print(f"Loading {model_key} from {model_path}…")

    local_model_path = resolve_hf_path(model_path)
    resolved_attn = rt.resolved_attn()
    quant_config = build_quantization_config(quantization, dtype)

    # Always point the processor at the locally-resolved codec. The MOSS
    # processors default ``codec_path`` to the bare repo id, which makes them
    # re-download the audio tokenizer into the *default* HF cache — a second
    # copy separate from the one our download step placed in the local dir.
    processor_kwargs: dict = {
        "trust_remote_code": True,
        "codec_path": resolve_hf_path(CODEC_MODEL_PATH),
    }

    processor = AutoProcessor.from_pretrained(local_model_path, **processor_kwargs)

    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
        processor.audio_tokenizer.eval()

    model_kwargs: dict = {"trust_remote_code": True, "torch_dtype": dtype}
    if resolved_attn:
        model_kwargs["attn_implementation"] = resolved_attn

    if quant_config is not None:
        # bitsandbytes places weights on the GPU itself; ``device_map`` is
        # required and a follow-up ``.to(device)`` must be skipped.
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = {"": device_str}
        print(f"  quantization: {quantization}")
        model = AutoModel.from_pretrained(local_model_path, **model_kwargs)
    else:
        model = AutoModel.from_pretrained(local_model_path, **model_kwargs).to(device)
    model.eval()

    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
    print(f"✓ {model_key} loaded")

    payload = (model, processor, device, sample_rate)
    cache_put(cache_key, payload)
    return payload


def load_realtime_model(device_str: str, attn_implementation: str):
    """Load the MOSS-TTS-Realtime inferencer + codec, sharing the residency budget."""
    cache_key = ("realtime", device_str, attn_implementation)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    evict_to_make_room(incoming=1)

    rt = _runtime(device_str, attn_implementation)
    device, dtype = rt.device, rt.dtype

    model_path = MODELS["realtime"]
    print(f"Loading realtime model from {model_path}…")

    local_model_path = resolve_hf_path(model_path)
    local_codec_path = resolve_hf_path(CODEC_MODEL_PATH)
    resolved_attn = rt.resolved_attn()

    # Ensure the vendored MOSS-TTS repo packages are importable.
    app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    moss_tts_root = os.path.join(app_root, "MOSS-TTS")
    moss_tts_realtime_dir = os.path.join(moss_tts_root, "moss_tts_realtime")
    for p in (moss_tts_root, moss_tts_realtime_dir):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime
    from inferencer import MossTTSRealtimeInference

    model_kwargs = {"torch_dtype": dtype}
    if resolved_attn:
        model_kwargs["attn_implementation"] = resolved_attn

    model = MossTTSRealtime.from_pretrained(local_model_path, **model_kwargs).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    # The codec must stay float32: it internally casts tensors to .float() for
    # numerical stability, then feeds them back into conv/linear layers — bf16
    # weights would cause dtype mismatches.
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
    payload = (inferencer, codec, device, 24000)
    cache_put(cache_key, payload)
    return payload


# ---------------------------------------------------------------------------
# Shared generation
# ---------------------------------------------------------------------------

def generate_and_decode(
    model,
    processor,
    device: torch.device,
    conversations,
    mode: str,
    sampling: Sampling,
    max_new_tokens: int,
) -> "np.ndarray":  # noqa: F821 - numpy imported lazily via audio_to_int16
    """Run a single batched generation and decode it to an int16 waveform.

    This is the one true generation path for the non-streaming tabs. It uses
    ``inference_mode`` (lighter than ``no_grad``) and frees the KV cache /
    activation buffers afterwards so long runs don't pin VRAM.
    """
    batch = processor(conversations, mode=mode)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    outputs = None
    try:
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                **sampling.as_generate_kwargs(),
            )

        messages = processor.decode(outputs)
        if not messages or messages[0] is None:
            raise RuntimeError("The model did not return a decodable audio result.")

        return audio_to_int16(messages[0].audio_codes_list[0])
    finally:
        del input_ids, attention_mask, outputs
        free_after_generation(device)

from pathlib import Path

import torch

# Disable the broken cuDNN SDPA backend; keep other backends as fallbacks.
# Applied at import time so every module gets the same settings.
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "tts": "OpenMOSS-Team/MOSS-TTS",
    "tts_local": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
    "ttsd": "OpenMOSS-Team/MOSS-TTSD-v1.0",
    "ttsd_v0.7": "OpenMOSS-Team/MOSS-TTSD-v0.7",
    "ttsd_v0.5": "OpenMOSS-Team/MOSS-TTSD-v0.5",
    "ttsd_v0": "OpenMOSS-Team/MOSS-TTSD-v0",
    "voice_gen": "OpenMOSS-Team/MOSS-VoiceGenerator",
    "sound_effect": "OpenMOSS-Team/MOSS-SoundEffect",
    "realtime": "OpenMOSS-Team/MOSS-TTS-Realtime",
}

# MOSS-Speech uses a separate codec and different processor API
SPEECH_MODEL_PATH = "OpenMOSS-Team/MOSS-Speech"
SPEECH_CODEC_PATH = "OpenMOSS-Team/MOSS-Speech-Codec"

# TTSD version display names → model keys
TTSD_VERSIONS = {
    "MOSS-TTSD v1.0 (8B) — recommended": "ttsd",
    "MOSS-TTSD v0.7 (2B)": "ttsd_v0.7",
    "MOSS-TTSD v0.5 (2B)": "ttsd_v0.5",
    "MOSS-TTSD v0 (2B)": "ttsd_v0",
}

CODEC_MODEL_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

# ---------------------------------------------------------------------------
# Audio / generation constants
# ---------------------------------------------------------------------------

# The audio tokenizer produces 12.5 tokens per second of audio
TOKENS_PER_SECOND = 12.5

# Reference clips longer than this are truncated to prevent O(L²) OOM
MAX_REFERENCE_DURATION_SEC = 30.0

# ---------------------------------------------------------------------------
# TTS continuation modes
# ---------------------------------------------------------------------------

CONTINUATION_NOTICE = (
    "Continuation mode is active. Make sure the reference audio transcript "
    "is prepended to the input text."
)
MODE_CLONE = "Clone"
MODE_CONTINUE = "Continuation"
MODE_CONTINUE_CLONE = "Continuation + Clone"

# ---------------------------------------------------------------------------
# Duration estimation (tokens per character, by language)
# ---------------------------------------------------------------------------

ZH_TOKENS_PER_CHAR = 3.098411951313033
EN_TOKENS_PER_CHAR = 0.8673376262755219

# ---------------------------------------------------------------------------
# Example asset paths (mirrors the HF Space layout)
# ---------------------------------------------------------------------------

REFERENCE_AUDIO_DIR = Path(__file__).resolve().parent / "assets" / "audio"
EXAMPLE_TEXTS_JSONL_PATH = (
    Path(__file__).resolve().parent / "assets" / "text" / "moss_tts_example_texts.jsonl"
)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

PRELOAD_ENV_VAR = "MOSS_TTS_PRELOAD_AT_STARTUP"

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
    "tts": "OpenMOSS-Team/MOSS-TTS-v1.5",
    "tts_local": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
    "ttsd": "OpenMOSS-Team/MOSS-TTSD-v1.0",
    "voice_gen": "OpenMOSS-Team/MOSS-VoiceGenerator",
    "sound_effect": "OpenMOSS-Team/MOSS-SoundEffect",
    "realtime": "OpenMOSS-Team/MOSS-TTS-Realtime",
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

# TTS tab model variant labels (must match Gradio Radio choices)
TTS_VARIANT_V15 = "MOSS-TTS-v1.5 (8B)"
TTS_VARIANT_LOCAL = "MOSS-TTS-Local (1.7B)"

# MOSS-TTS-v1.5 language tags (English names as expected by build_user_message)
TTS_LANGUAGE_AUTO = "Auto (no tag)"
TTS_LANGUAGE_CHOICES = [
    TTS_LANGUAGE_AUTO,
    "Chinese",
    "Cantonese",
    "English",
    "Arabic",
    "Czech",
    "Danish",
    "Dutch",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hebrew",
    "Hindi",
    "Hungarian",
    "Italian",
    "Japanese",
    "Korean",
    "Macedonian",
    "Malay",
    "Persian (Farsi)",
    "Polish",
    "Portuguese",
    "Romanian",
    "Russian",
    "Spanish",
    "Swahili",
    "Swedish",
    "Tagalog",
    "Thai",
    "Turkish",
    "Vietnamese",
]


def resolve_tts_language(choice: str):
    """Map UI language choice to processor language tag, or None to omit."""
    if not choice or choice == TTS_LANGUAGE_AUTO:
        return None
    return choice

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

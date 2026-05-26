"""Shared utilities: env parsing, language detection, duration control,
TTS conversation building, and example-row loading."""

import os
import re
from pathlib import Path
from typing import Optional, Tuple

try:
    import orjson
    def _json_loads(b):
        return orjson.loads(b)
except ImportError:
    import json
    def _json_loads(b):
        return json.loads(b.decode() if isinstance(b, bytes) else b)

import gradio as gr

from config import (
    CONTINUATION_NOTICE,
    EN_TOKENS_PER_CHAR,
    EXAMPLE_TEXTS_JSONL_PATH,
    MODE_CLONE,
    MODE_CONTINUE,
    MODE_CONTINUE_CLONE,
    REFERENCE_AUDIO_DIR,
    ZH_TOKENS_PER_CHAR,
)

# ---------------------------------------------------------------------------
# Environment / CLI helpers
# ---------------------------------------------------------------------------

def parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_port(value: Optional[str], default: int) -> int:
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Language detection & duration estimation
# ---------------------------------------------------------------------------

def detect_text_language(text: str) -> str:
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_chars = len(re.findall(r"[A-Za-z]", text))
    if zh_chars == 0 and en_chars == 0:
        return "en"
    return "zh" if zh_chars >= en_chars else "en"


def supports_duration_control(mode_with_reference: str) -> bool:
    return mode_with_reference not in {MODE_CONTINUE, MODE_CONTINUE_CLONE}


def estimate_duration_tokens(text: str) -> Tuple[str, int, int, int]:
    normalized = text or ""
    effective_len = max(len(normalized), 1)
    language = detect_text_language(normalized)
    factor = ZH_TOKENS_PER_CHAR if language == "zh" else EN_TOKENS_PER_CHAR
    default_tokens = max(1, int(effective_len * factor))
    min_tokens = max(1, int(default_tokens * 0.5))
    max_tokens = max(min_tokens, int(default_tokens * 1.5))
    return language, default_tokens, min_tokens, max_tokens


def update_duration_controls(enabled: bool, text: str, current_tokens, mode_with_reference: str):
    """Return (slider_update, hint_text, checkbox_update) for Gradio reactivity."""
    if not supports_duration_control(mode_with_reference):
        return (
            gr.update(visible=False),
            "Duration control is disabled for Continuation modes.",
            gr.update(value=False, interactive=False),
        )
    checkbox_update = gr.update(interactive=True)
    if not enabled:
        return gr.update(visible=False), "Duration control is disabled.", checkbox_update

    language, default_tokens, min_tokens, max_tokens = estimate_duration_tokens(text)
    if current_tokens is None or int(current_tokens) == 1:
        slider_value = default_tokens
    else:
        slider_value = int(current_tokens)
        slider_value = max(min_tokens, min(max_tokens, slider_value))

    language_label = "Chinese" if language == "zh" else "English"
    hint = (
        f"Duration control enabled | detected language: {language_label} | "
        f"default={default_tokens}, range=[{min_tokens}, {max_tokens}]"
    )
    return (
        gr.update(visible=True, minimum=min_tokens, maximum=max_tokens, value=slider_value, step=1),
        hint,
        checkbox_update,
    )


# ---------------------------------------------------------------------------
# TTS mode hint & conversation builder
# ---------------------------------------------------------------------------

def format_pause_marker(duration: float) -> str:
    """Build a MOSS-TTS-v1.5 inline pause marker."""
    seconds = max(0.1, min(30.0, float(duration)))
    return f"[pause {seconds:.1f}s]"


def insert_pause_marker(text: str, duration: float) -> str:
    """Append a pause marker to the synthesis text (server-side fallback)."""
    marker = format_pause_marker(duration)
    return (text or "") + marker


def render_mode_hint(reference_audio: Optional[str], mode_with_reference: str) -> str:
    if not reference_audio:
        return "Current mode: **Direct Generation** (no reference audio uploaded)"
    if mode_with_reference == MODE_CLONE:
        return "Current mode: **Clone** (speaker timbre will be cloned from the reference audio)"
    return f"Current mode: **{mode_with_reference}**  \n> {CONTINUATION_NOTICE}"


def build_tts_conversation(
    text: str,
    reference_audio: Optional[str],
    mode_with_reference: str,
    expected_tokens: Optional[int],
    processor,
    language: Optional[str] = None,
):
    """Build the processor conversation list for any TTS mode.

    Returns (conversations, mode_str, mode_name).
    """
    user_kwargs = {"text": text}
    if language:
        user_kwargs["language"] = language
    if expected_tokens is not None:
        user_kwargs["tokens"] = int(expected_tokens)

    if not reference_audio:
        return [[processor.build_user_message(**user_kwargs)]], "generation", "Direct Generation"

    if mode_with_reference == MODE_CLONE:
        clone_kwargs = dict(user_kwargs, reference=[reference_audio])
        return [[processor.build_user_message(**clone_kwargs)]], "generation", MODE_CLONE

    if mode_with_reference == MODE_CONTINUE:
        conversations = [[
            processor.build_user_message(**user_kwargs),
            processor.build_assistant_message(audio_codes_list=[reference_audio]),
        ]]
        return conversations, "continuation", MODE_CONTINUE

    # Continuation + Clone
    continue_clone_kwargs = dict(user_kwargs, reference=[reference_audio])
    conversations = [[
        processor.build_user_message(**continue_clone_kwargs),
        processor.build_assistant_message(audio_codes_list=[reference_audio]),
    ]]
    return conversations, "continuation", MODE_CONTINUE_CLONE


# ---------------------------------------------------------------------------
# Example rows (loaded once at import time)
# ---------------------------------------------------------------------------

def _parse_example_id(example_id: str) -> Optional[Tuple[str, int]]:
    matched = re.fullmatch(r"(zh|en)/(\d+)", (example_id or "").strip())
    if matched is None:
        return None
    return matched.group(1), int(matched.group(2))


def _resolve_reference_audio_path(language: str, index: int) -> Optional[Path]:
    for stem in [f"reference_{language}_{index}"]:
        for ext in (".wav", ".mp3"):
            audio_path = REFERENCE_AUDIO_DIR / f"{stem}{ext}"
            if audio_path.exists():
                return audio_path
    return None


def build_example_rows() -> list:
    rows = []
    if not EXAMPLE_TEXTS_JSONL_PATH.exists():
        return rows
    try:
        with open(EXAMPLE_TEXTS_JSONL_PATH, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = _json_loads(line)
                parsed = _parse_example_id(sample.get("id", ""))
                if parsed is None:
                    continue
                language, index = parsed
                text = str(sample.get("text", "")).strip()
                audio_path = _resolve_reference_audio_path(language, index)
                if audio_path is None:
                    continue
                rows.append((sample.get("role", ""), str(audio_path), text))
    except Exception as e:
        print(f"⚠️  Could not load example rows: {e}")
    return rows


EXAMPLE_ROWS = build_example_rows()

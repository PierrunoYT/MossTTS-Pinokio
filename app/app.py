"""
MOSS-TTS Unified Interface
===========================
All-in-one Gradio UI combining:
- MOSS-TTS (Main TTS with voice cloning)
- MOSS-TTSD (Dialogue generation)
- MOSS-VoiceGenerator (Voice design from text prompts)
- MOSS-SoundEffect (Sound effect generation)
- MOSS-TTS-Realtime (Low-latency streaming TTS for voice agents)

Usage:
    python app.py [--device cuda:0] [--port 7860] [--share]
"""

import argparse
import functools
import importlib.util
import os
import re
import sys
import tempfile
import time
import traceback
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
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

# ============================================================================
# Configuration
# ============================================================================

# Disable problematic cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# Model paths
MODELS = {
    "tts": "OpenMOSS-Team/MOSS-TTS",
    "tts_local": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
    "ttsd": "OpenMOSS-Team/MOSS-TTSD-v1.0",
    "voice_gen": "OpenMOSS-Team/MOSS-VoiceGenerator",
    "sound_effect": "OpenMOSS-Team/MOSS-SoundEffect",
    "realtime": "OpenMOSS-Team/MOSS-TTS-Realtime",
}

CODEC_MODEL_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer"

# Audio tokenizer produces 12.5 tokens per second of audio
TOKENS_PER_SECOND = 12.5

# Max reference audio duration to avoid OOM in the audio tokenizer's self-attention
MAX_REFERENCE_DURATION_SEC = 30.0

# TTS continuation modes
CONTINUATION_NOTICE = (
    "Continuation mode is active. Make sure the reference audio transcript is prepended to the input text."
)
MODE_CLONE = "Clone"
MODE_CONTINUE = "Continuation"
MODE_CONTINUE_CLONE = "Continuation + Clone"

# Duration estimation constants (tokens per character by language)
ZH_TOKENS_PER_CHAR = 3.098411951313033
EN_TOKENS_PER_CHAR = 0.8673376262755219

# Example asset paths (mirrors the HF Space layout)
REFERENCE_AUDIO_DIR = Path(__file__).resolve().parent / "assets" / "audio"
EXAMPLE_TEXTS_JSONL_PATH = (
    Path(__file__).resolve().parent / "assets" / "text" / "moss_tts_example_texts.jsonl"
)

PRELOAD_ENV_VAR = "MOSS_TTS_PRELOAD_AT_STARTUP"

# ============================================================================
# Helper Functions
# ============================================================================

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
):
    user_kwargs = {"text": text}
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


# ============================================================================
# Model Loading
# ============================================================================

def resolve_attn_implementation(requested: str, device: torch.device, dtype: torch.dtype) -> Optional[str]:
    """Resolve the best attention implementation for the given device and dtype."""
    requested_norm = (requested or "").strip().lower()

    if requested_norm in {"none"}:
        return None

    if requested_norm not in {"", "auto"}:
        return requested

    # Prefer FlashAttention 2 when available
    if (
        device.type == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return "flash_attention_2"

    # CUDA fallback
    if device.type == "cuda":
        return "sdpa"

    # CPU fallback
    return "eager"


def _resolve_hf_path(repo_id: str) -> str:
    """Resolve a HuggingFace repo ID to a local snapshot path on Windows.

    The model's custom processor code converts the path with ``Path()``,
    which on Windows turns the ``/`` in repo IDs (e.g. ``Org/Model``) into
    backslashes, producing an invalid HuggingFace repo ID.  Pre-downloading
    with ``snapshot_download`` gives us a real local directory path where
    backslashes are expected.

    Retries up to 5 times on network errors (common on Windows for large
    multi-GB downloads). Each interrupted attempt resumes from where it
    left off thanks to HuggingFace Hub's local cache.
    """
    if sys.platform == "win32" and "/" in repo_id and not os.path.isdir(repo_id):
        import time
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
                    f"Retrying in {wait}s... (attempt {attempt}/{max_attempts})"
                )
                time.sleep(wait)
    return repo_id


def _truncate_reference_audio(audio_path: str, max_duration: float = MAX_REFERENCE_DURATION_SEC) -> str:
    """Truncate reference audio to max_duration seconds to prevent OOM in the tokenizer.

    The audio tokenizer's self-attention computes an O(L²) positional matrix;
    very long reference clips cause out-of-memory errors on consumer GPUs.
    Returns the original path if the file is already short enough, otherwise
    writes a truncated copy to a temp file and returns its path.
    """
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


@functools.lru_cache(maxsize=6)
def load_model(model_key: str, device_str: str, attn_implementation: str):
    """Load and cache a model."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    model_path = MODELS[model_key]
    print(f"Loading {model_key} from {model_path}...")
    
    # On Windows, resolve HF repo IDs to local paths to avoid backslash issues
    # in model custom code that uses Path() on the repo ID string.
    local_model_path = _resolve_hf_path(model_path)
    
    resolved_attn = resolve_attn_implementation(attn_implementation, device, dtype)
    
    # Load processor
    processor_kwargs = {"trust_remote_code": True}
    if model_key == "ttsd":
        processor_kwargs["codec_path"] = _resolve_hf_path(CODEC_MODEL_PATH)
    
    processor = AutoProcessor.from_pretrained(local_model_path, **processor_kwargs)
    
    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
        processor.audio_tokenizer.eval()
    
    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if resolved_attn:
        model_kwargs["attn_implementation"] = resolved_attn
    
    model = AutoModel.from_pretrained(local_model_path, **model_kwargs).to(device)
    model.eval()
    
    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
    
    print(f"✓ {model_key} loaded successfully")
    return model, processor, device, sample_rate


# ============================================================================
# TAB 1: MOSS-TTS (Main TTS with Voice Cloning)
# ============================================================================

def run_tts_inference(
    text: str,
    reference_audio: Optional[str],
    mode_with_reference: str,
    duration_control_enabled: bool,
    duration_tokens: int,
    model_variant: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    device: str,
    attn_implementation: str,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """Run MOSS-TTS inference with support for Clone, Continuation, and Continuation+Clone modes."""
    started_at = time.monotonic()
    _ref_tmp = None
    try:
        if not text or not text.strip():
            return None, "❌ Error: Please enter text to synthesize"

        model_key = "tts_local" if model_variant == "MOSS-TTS-Local (1.7B)" else "tts"
        model, processor, dev, sample_rate = load_model(model_key, device, attn_implementation)

        # Truncate reference audio to avoid OOM if needed
        resolved_ref = None
        if reference_audio:
            _ref_tmp = _truncate_reference_audio(reference_audio)
            resolved_ref = _ref_tmp

        # Compute expected tokens from the smart duration control
        duration_enabled = bool(duration_control_enabled and supports_duration_control(mode_with_reference))
        expected_tokens = int(duration_tokens) if duration_enabled else None

        # Build conversation (handles all four modes)
        conversations, mode, mode_name = build_tts_conversation(
            text=text,
            reference_audio=resolved_ref,
            mode_with_reference=mode_with_reference,
            expected_tokens=expected_tokens,
            processor=processor,
        )

        batch = processor(conversations, mode=mode)
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                audio_temperature=float(temperature),
                audio_top_p=float(top_p),
                audio_top_k=int(top_k),
                audio_repetition_penalty=float(repetition_penalty),
            )

        messages = processor.decode(outputs)
        if not messages or messages[0] is None:
            raise RuntimeError("The model did not return a decodable audio result.")

        audio = messages[0].audio_codes_list[0]
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().float().cpu().numpy()
        else:
            audio_np = np.asarray(audio, dtype=np.float32)
        if audio_np.ndim > 1:
            audio_np = audio_np.reshape(-1)
        audio_np = audio_np.astype(np.float32, copy=False)

        if _ref_tmp and _ref_tmp != reference_audio:
            try:
                os.unlink(_ref_tmp)
            except OSError:
                pass

        elapsed = time.monotonic() - started_at
        status = (
            f"✅ Done | mode: {mode_name} | elapsed: {elapsed:.2f}s | "
            f"max_new_tokens={max_new_tokens}, "
            f"expected_tokens={expected_tokens if expected_tokens is not None else 'off'}, "
            f"temperature={temperature:.2f}, top_p={top_p:.2f}, top_k={top_k}"
        )
        return (sample_rate, audio_np), status

    except Exception as e:
        if _ref_tmp and _ref_tmp != reference_audio:
            try:
                os.unlink(_ref_tmp)
            except OSError:
                pass
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def build_tts_tab(args):
    """Build the MOSS-TTS tab."""
    with gr.Column():
        gr.Markdown("### 🎙️ MOSS-TTS - High-Quality Voice Cloning")
        gr.Markdown("Generate speech with or without reference audio. Supports voice cloning and continuation modes.")

        tts_model_variant = gr.Radio(
            choices=["MOSS-TTS (8B)", "MOSS-TTS-Local (1.7B)"],
            value="MOSS-TTS (8B)",
            label="Model Variant",
            info="8B: best zero-shot cloning & long-form stability. 1.7B Local: highest speaker similarity score, lighter VRAM.",
        )

        with gr.Row():
            with gr.Column(scale=3):
                tts_text = gr.Textbox(
                    label="Text to Synthesize",
                    lines=8,
                    placeholder="Enter text here. In Continuation modes, prepend the reference audio transcript.",
                )
                tts_reference = gr.Audio(
                    label="Reference Audio (Optional)",
                    type="filepath",
                )
                tts_mode = gr.Radio(
                    choices=[MODE_CLONE, MODE_CONTINUE, MODE_CONTINUE_CLONE],
                    value=MODE_CLONE,
                    label="Mode with Reference Audio",
                    info="If no reference audio is uploaded, Direct Generation is used automatically.",
                )
                tts_mode_hint = gr.Markdown(render_mode_hint(None, MODE_CLONE))

                tts_duration_enabled = gr.Checkbox(
                    value=False,
                    label="Enable Duration Control (Expected Audio Tokens)",
                )
                tts_duration_tokens = gr.Slider(
                    minimum=1, maximum=1, step=1, value=1,
                    label="expected_tokens",
                    visible=False,
                )
                tts_duration_hint = gr.Markdown("Duration control is disabled.")

                with gr.Accordion("Sampling Parameters", open=False):
                    tts_temp = gr.Slider(0.1, 3.0, value=1.7, step=0.05, label="Temperature")
                    tts_top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.01, label="Top P")
                    tts_top_k = gr.Slider(1, 200, value=25, step=1, label="Top K")
                    tts_rep_penalty = gr.Slider(0.8, 2.0, value=1.0, step=0.05, label="Repetition Penalty")
                    tts_max_tokens = gr.Slider(256, 8192, value=4096, step=128, label="Max New Tokens")

                tts_generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")

            with gr.Column(scale=2):
                tts_output = gr.Audio(label="Generated Audio")
                tts_status = gr.Textbox(label="Status", lines=4, interactive=False)

                if EXAMPLE_ROWS:
                    tts_examples = gr.Dataframe(
                        headers=["Reference Speaker", "Example Text"],
                        value=[[name, text] for name, _, text in EXAMPLE_ROWS],
                        datatype=["str", "str"],
                        row_count=(len(EXAMPLE_ROWS), "fixed"),
                        col_count=(2, "fixed"),
                        interactive=False,
                        wrap=True,
                        label="Examples — click a row to fill inputs",
                    )

        # Mode hint reactivity
        tts_reference.change(
            fn=render_mode_hint,
            inputs=[tts_reference, tts_mode],
            outputs=[tts_mode_hint],
        )
        tts_mode.change(
            fn=render_mode_hint,
            inputs=[tts_reference, tts_mode],
            outputs=[tts_mode_hint],
        )

        # Duration control reactivity
        for trigger in [tts_duration_enabled, tts_text, tts_mode]:
            trigger.change(
                fn=update_duration_controls,
                inputs=[tts_duration_enabled, tts_text, tts_duration_tokens, tts_mode],
                outputs=[tts_duration_tokens, tts_duration_hint, tts_duration_enabled],
            )

        # Example row click handler
        if EXAMPLE_ROWS:
            def _apply_example(mode, dur_enabled, dur_tokens, evt: gr.SelectData):
                if evt is None or evt.index is None:
                    return [gr.update()] * 6
                row_idx = int(evt.index[0]) if isinstance(evt.index, (tuple, list)) else int(evt.index)
                if row_idx < 0 or row_idx >= len(EXAMPLE_ROWS):
                    return [gr.update()] * 6
                _, audio_path, example_text = EXAMPLE_ROWS[row_idx]
                dur_slider, dur_hint, dur_checkbox = update_duration_controls(
                    dur_enabled, example_text, dur_tokens, mode
                )
                return audio_path, example_text, render_mode_hint(audio_path, mode), dur_slider, dur_hint, dur_checkbox

            tts_examples.select(
                fn=_apply_example,
                inputs=[tts_mode, tts_duration_enabled, tts_duration_tokens],
                outputs=[tts_reference, tts_text, tts_mode_hint, tts_duration_tokens, tts_duration_hint, tts_duration_enabled],
            )

        tts_generate_btn.click(
            fn=lambda *x: run_tts_inference(*x, args.device, args.attn_implementation),
            inputs=[
                tts_text, tts_reference, tts_mode,
                tts_duration_enabled, tts_duration_tokens,
                tts_model_variant,
                tts_temp, tts_top_p, tts_top_k, tts_rep_penalty, tts_max_tokens,
            ],
            outputs=[tts_output, tts_status],
        )


# ============================================================================
# TAB 2: MOSS-TTSD (Dialogue Generation)
# ============================================================================

def run_ttsd_inference(
    script_text: str,
    num_speakers: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    device: str,
    attn_implementation: str,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """Run MOSS-TTSD dialogue generation."""
    try:
        if not script_text or not script_text.strip():
            return None, "❌ Error: Please enter dialogue script"

        # Validate speaker tags
        used_speakers = set(int(m) for m in re.findall(r'\[S(\d+)\]', script_text))
        if not used_speakers:
            return None, "❌ Error: Dialogue must include speaker tags like [S1], [S2], ..."
        if max(used_speakers) > num_speakers:
            return None, f"❌ Error: Script uses [S{max(used_speakers)}] but only {num_speakers} speaker(s) configured"

        model, processor, dev, sample_rate = load_model("ttsd", device, attn_implementation)

        # Build conversation using the processor's message builder
        conversation = [processor.build_user_message(text=script_text)]

        # Process inputs
        batch = processor(conversation, mode="generation", num_speakers=num_speakers)
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                audio_temperature=temperature,
                audio_top_p=top_p,
                audio_top_k=top_k,
                audio_repetition_penalty=repetition_penalty,
            )

        # Decode
        messages = processor.decode(outputs)
        if messages and len(messages) > 0:
            audio = messages[0].audio_codes_list[0]
            audio_np = audio.cpu().numpy()
            return (sample_rate, audio_np), "✅ Dialogue generation completed!"

        return None, "❌ Error: No audio generated"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def build_ttsd_tab(args):
    """Build the MOSS-TTSD tab."""
    with gr.Column():
        gr.Markdown("### 💬 MOSS-TTSD - Multi-Speaker Dialogue Generation")
        gr.Markdown("Generate expressive multi-speaker dialogues from scripts.")
        
        with gr.Row():
            with gr.Column(scale=1):
                ttsd_script = gr.Textbox(
                    label="Dialogue Script",
                    lines=10,
                    placeholder="[S1] Hello, how are you doing today?\n[S2] I'm doing great, thanks for asking!\n[S1] That's wonderful to hear.",
                    info="Use [S1], [S2], ... tags to label each speaker's turn.",
                )
                ttsd_num_speakers = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    label="Number of Speakers",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    ttsd_temp = gr.Slider(0.1, 3.0, value=1.1, step=0.05, label="Temperature")
                    ttsd_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.01, label="Top P")
                    ttsd_top_k = gr.Slider(1, 200, value=50, step=1, label="Top K")
                    ttsd_rep_penalty = gr.Slider(0.8, 2.0, value=1.1, step=0.05, label="Repetition Penalty")
                    ttsd_max_tokens = gr.Slider(256, 8192, value=2000, step=128, label="Max New Tokens")

                ttsd_generate_btn = gr.Button("🎭 Generate Dialogue", variant="primary", size="lg")

            with gr.Column(scale=1):
                ttsd_output = gr.Audio(label="Generated Dialogue")
                ttsd_status = gr.Textbox(label="Status", lines=3, interactive=False)

        ttsd_generate_btn.click(
            fn=lambda *x: run_ttsd_inference(*x, args.device, args.attn_implementation),
            inputs=[ttsd_script, ttsd_num_speakers, ttsd_temp, ttsd_top_p, ttsd_top_k, ttsd_rep_penalty, ttsd_max_tokens],
            outputs=[ttsd_output, ttsd_status],
        )


# ============================================================================
# TAB 3: MOSS-VoiceGenerator (Voice Design)
# ============================================================================

def run_voice_gen_inference(
    instruction: str,
    text: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    device: str,
    attn_implementation: str,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """Run MOSS-VoiceGenerator inference."""
    try:
        if not instruction or not instruction.strip():
            return None, "❌ Error: Please enter voice description"
        if not text or not text.strip():
            return None, "❌ Error: Please enter text to synthesize"

        model, processor, dev, sample_rate = load_model("voice_gen", device, attn_implementation)

        # Build conversation
        conversation = [processor.build_user_message(instruction=instruction, text=text)]

        # Process inputs
        batch = processor(conversation, mode="generation")
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                audio_temperature=temperature,
                audio_top_p=top_p,
                audio_top_k=top_k,
                audio_repetition_penalty=repetition_penalty,
            )

        # Decode
        messages = processor.decode(outputs)
        if messages and len(messages) > 0:
            audio = messages[0].audio_codes_list[0]
            audio_np = audio.cpu().numpy()
            return (sample_rate, audio_np), "✅ Voice generation completed!"

        return None, "❌ Error: No audio generated"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def build_voice_gen_tab(args):
    """Build the MOSS-VoiceGenerator tab."""
    with gr.Column():
        gr.Markdown("### 🎨 MOSS-VoiceGenerator - Design Voices from Text")
        gr.Markdown("Create unique voices by describing them in natural language.")
        
        with gr.Row():
            with gr.Column(scale=1):
                vg_instruction = gr.Textbox(
                    label="Voice Description",
                    lines=4,
                    placeholder="Describe the voice you want (e.g., 'A young female with a cheerful and energetic tone')...",
                )
                vg_text = gr.Textbox(
                    label="Text to Synthesize",
                    lines=6,
                    placeholder="Enter the text for the voice to speak...",
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    vg_temp = gr.Slider(0.1, 3.0, value=1.5, step=0.05, label="Temperature")
                    vg_top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.01, label="Top P")
                    vg_top_k = gr.Slider(1, 200, value=25, step=1, label="Top K")
                    vg_rep_penalty = gr.Slider(0.8, 2.0, value=1.0, step=0.05, label="Repetition Penalty")
                    vg_max_tokens = gr.Slider(256, 8192, value=4096, step=128, label="Max New Tokens")

                vg_generate_btn = gr.Button("✨ Generate Voice", variant="primary", size="lg")

            with gr.Column(scale=1):
                vg_output = gr.Audio(label="Generated Audio")
                vg_status = gr.Textbox(label="Status", lines=3, interactive=False)

                gr.Markdown("**Example Descriptions:**")
                gr.Markdown("- A middle-aged male with a deep, authoritative voice\n- A young child with a playful tone\n- An elderly woman with a warm, gentle voice")

        vg_generate_btn.click(
            fn=lambda *x: run_voice_gen_inference(*x, args.device, args.attn_implementation),
            inputs=[vg_instruction, vg_text, vg_temp, vg_top_p, vg_top_k, vg_rep_penalty, vg_max_tokens],
            outputs=[vg_output, vg_status],
        )


# ============================================================================
# TAB 4: MOSS-SoundEffect (Sound Generation)
# ============================================================================

def run_sound_effect_inference(
    description: str,
    duration_seconds: float,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    device: str,
    attn_implementation: str,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """Run MOSS-SoundEffect inference."""
    try:
        if not description or not description.strip():
            return None, "❌ Error: Please enter sound description"

        model, processor, dev, sample_rate = load_model("sound_effect", device, attn_implementation)

        # Convert duration to tokens (12.5 tokens/second)
        expected_tokens = max(1, int(duration_seconds * TOKENS_PER_SECOND))

        # Build conversation using build_user_message with duration control
        conversation = [processor.build_user_message(text=description, tokens=expected_tokens)]

        # Process inputs
        batch = processor(conversation, mode="generation")
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                audio_temperature=temperature,
                audio_top_p=top_p,
                audio_top_k=top_k,
                audio_repetition_penalty=repetition_penalty,
            )

        # Decode
        messages = processor.decode(outputs)
        if messages and len(messages) > 0:
            audio = messages[0].audio_codes_list[0]
            audio_np = audio.cpu().numpy()
            return (sample_rate, audio_np), "✅ Sound effect generated!"

        return None, "❌ Error: No audio generated"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def build_sound_effect_tab(args):
    """Build the MOSS-SoundEffect tab."""
    with gr.Column():
        gr.Markdown("### 🔊 MOSS-SoundEffect - Generate Sound Effects")
        gr.Markdown("Create sound effects and environmental audio from text descriptions.")
        
        with gr.Row():
            with gr.Column(scale=1):
                se_description = gr.Textbox(
                    label="Sound Description",
                    lines=4,
                    placeholder="Describe the sound you want (e.g., 'Thunder and rain', 'City traffic', 'Forest birds')...",
                )
                se_duration = gr.Slider(
                    1, 60, value=10, step=1,
                    label="Duration (seconds)",
                    info="Target length of the generated sound effect.",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    se_temp = gr.Slider(0.1, 3.0, value=1.5, step=0.05, label="Temperature")
                    se_top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.01, label="Top P")
                    se_top_k = gr.Slider(1, 200, value=25, step=1, label="Top K")
                    se_rep_penalty = gr.Slider(0.8, 2.0, value=1.0, step=0.05, label="Repetition Penalty")
                    se_max_tokens = gr.Slider(256, 8192, value=4096, step=128, label="Max New Tokens")

                se_generate_btn = gr.Button("🎵 Generate Sound", variant="primary", size="lg")

            with gr.Column(scale=1):
                se_output = gr.Audio(label="Generated Sound")
                se_status = gr.Textbox(label="Status", lines=3, interactive=False)

                gr.Markdown("**Example Sounds:**")
                gr.Markdown("- Ocean waves crashing on the beach\n- Busy city street with traffic\n- Birds chirping in a forest\n- Thunderstorm with heavy rain")

        se_generate_btn.click(
            fn=lambda *x: run_sound_effect_inference(*x, args.device, args.attn_implementation),
            inputs=[se_description, se_duration, se_temp, se_top_p, se_top_k, se_rep_penalty, se_max_tokens],
            outputs=[se_output, se_status],
        )


# ============================================================================
# TAB 5: MOSS-TTS-Realtime (Low-Latency Streaming TTS)
# ============================================================================

def run_realtime_inference(
    text: str,
    reference_audio: Optional[str],
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    device: str,
    attn_implementation: str,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """Run MOSS-TTS-Realtime inference."""
    try:
        if not text or not text.strip():
            return None, "❌ Error: Please enter text to synthesize"

        model, processor, dev, sample_rate = load_model("realtime", device, attn_implementation)

        msg_kwargs = {"text": text}
        if reference_audio:
            msg_kwargs["reference"] = [reference_audio]

        conversation = [processor.build_user_message(**msg_kwargs)]

        batch = processor(conversation, mode="generation")
        input_ids = batch["input_ids"].to(dev)
        attention_mask = batch["attention_mask"].to(dev)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                audio_temperature=temperature,
                audio_top_p=top_p,
                audio_top_k=top_k,
            )

        messages = processor.decode(outputs)
        if messages and len(messages) > 0:
            audio = messages[0].audio_codes_list[0]
            audio_np = audio.cpu().numpy()
            return (sample_rate, audio_np), "✅ Realtime generation completed!"

        return None, "❌ Error: No audio generated"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def build_realtime_tab(args):
    """Build the MOSS-TTS-Realtime tab."""
    with gr.Column():
        gr.Markdown("### ⚡ MOSS-TTS-Realtime - Low-Latency Voice Agent TTS")
        gr.Markdown(
            "1.7B streaming model optimised for real-time voice agents. "
            "Achieves ~180 ms TTFB after warm-up. "
            "Optionally supply a reference audio to anchor the speaker voice."
        )

        with gr.Row():
            with gr.Column(scale=1):
                rt_text = gr.Textbox(
                    label="Text to Synthesize",
                    lines=6,
                    placeholder="Enter text here...",
                )
                rt_reference = gr.Audio(
                    label="Reference Audio (Optional)",
                    type="filepath",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    rt_temp = gr.Slider(0.1, 3.0, value=1.0, step=0.05, label="Temperature")
                    rt_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.01, label="Top P")
                    rt_top_k = gr.Slider(1, 200, value=50, step=1, label="Top K")
                    rt_max_tokens = gr.Slider(256, 4096, value=2048, step=128, label="Max New Tokens")

                rt_generate_btn = gr.Button("⚡ Generate (Realtime)", variant="primary", size="lg")

            with gr.Column(scale=1):
                rt_output = gr.Audio(label="Generated Audio")
                rt_status = gr.Textbox(label="Status", lines=3, interactive=False)

                gr.Markdown("**About this model:**")
                gr.Markdown(
                    "- Architecture: MossTTSRealtime (1.7B)\n"
                    "- TTFB: ~180 ms (after warm-up)\n"
                    "- Ideal for voice agents paired with LLMs\n"
                    "- Supports multi-turn context via reference audio"
                )

        rt_generate_btn.click(
            fn=lambda *x: run_realtime_inference(*x, args.device, args.attn_implementation),
            inputs=[rt_text, rt_reference, rt_temp, rt_top_p, rt_top_k, rt_max_tokens],
            outputs=[rt_output, rt_status],
        )


# ============================================================================
# TAB 6: Info & About
# ============================================================================

def build_info_tab():
    """Build the info/about tab."""
    with gr.Column():
        gr.Markdown("""
        # 🎵 MOSS-TTS Unified Interface
        
        Welcome to the all-in-one interface for the MOSS-TTS Family of models!
        
        ## 📚 Available Models
        
        ### 🎙️ MOSS-TTS
        High-fidelity text-to-speech with zero-shot voice cloning. Upload a reference audio to clone any voice!
        Choose between **MOSS-TTS (8B)** for best long-form stability and **MOSS-TTS-Local (1.7B)** for the highest speaker similarity score with lower VRAM usage.
        
        ### 💬 MOSS-TTSD
        Multi-speaker dialogue generation for creating realistic conversations with different voices.
        
        ### 🎨 MOSS-VoiceGenerator
        Design custom voices from text descriptions without needing reference audio.
        
        ### 🔊 MOSS-SoundEffect
        Generate environmental sounds and effects from text descriptions.
        
        ### ⚡ MOSS-TTS-Realtime
        Low-latency streaming TTS optimised for real-time voice agents. Achieves ~180 ms TTFB after warm-up (1.7B model).
        
        ## 🚀 Quick Start
        
        1. **Choose a tab** for the model you want to use
        2. **Enter your text** or description
        3. **Adjust settings** if needed (optional)
        4. **Click Generate** and wait for the result
        
        ## ⚙️ Tips
        
        - For better quality, adjust the **Temperature** (higher = more creative, lower = more stable)
        - Use **reference audio** in MOSS-TTS for voice cloning
        - Be descriptive in voice/sound descriptions for better results
        - Generation time depends on text length and your hardware
        
        ## 📖 Learn More
        
        - [GitHub Repository](https://github.com/OpenMOSS/MOSS-TTS)
        - [Model Cards on HuggingFace](https://huggingface.co/collections/OpenMOSS-Team/moss-tts)
        - [MOSI.AI Website](https://mosi.cn/models/moss-tts)
        
        ## 📝 License
        
        All models are released under Apache 2.0 License.
        
        ---
        
        **Note:** First-time generation may take longer as models are downloaded and loaded.
        """)


# ============================================================================
# Main Application
# ============================================================================

def build_unified_interface(args):
    """Build the unified Gradio interface with all tabs."""
    
    custom_css = """
    .app-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .app-header h1 {
        margin: 0;
        font-size: 2.5em;
    }
    .app-header p {
        margin: 10px 0 0 0;
        font-size: 1.2em;
        opacity: 0.9;
    }
    """
    
    with gr.Blocks(title="MOSS-TTS Unified Interface", css=custom_css, theme=gr.themes.Soft()) as app:
        
        gr.HTML("""
        <div class="app-header">
            <h1>🎵 MOSS-TTS Family</h1>
            <p>Unified Interface for All Models</p>
        </div>
        """)
        
        with gr.Tabs():
            with gr.Tab("🎙️ TTS - Voice Cloning"):
                build_tts_tab(args)

            with gr.Tab("💬 TTSD - Dialogue"):
                build_ttsd_tab(args)

            with gr.Tab("🎨 Voice Generator"):
                build_voice_gen_tab(args)

            with gr.Tab("🔊 Sound Effects"):
                build_sound_effect_tab(args)

            with gr.Tab("⚡ Realtime TTS"):
                build_realtime_tab(args)

            with gr.Tab("ℹ️ About"):
                build_info_tab()
        
        gr.Markdown("---")
        gr.Markdown("Built with ❤️ by the OpenMOSS Team | Powered by Gradio")
    
    return app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MOSS-TTS Unified Interface")
    parser.add_argument("--model_path", type=str, default=MODELS["tts"])
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0, cpu)")
    parser.add_argument("--attn_implementation", type=str, default="auto", help="Attention implementation")
    _default_host = "127.0.0.1" if sys.platform == "win32" else "0.0.0.0"
    parser.add_argument("--host", type=str, default=_default_host, help="Host to bind to")
    parser.add_argument(
        "--port",
        type=int,
        default=parse_port(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT")), 7860),
    )
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    args = parser.parse_args()

    args.host = os.getenv("GRADIO_SERVER_NAME", args.host)
    args.port = parse_port(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT")), args.port)

    # Resolve attention implementation
    runtime_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runtime_dtype = torch.bfloat16 if runtime_device.type == "cuda" else torch.float32
    args.attn_implementation = resolve_attn_implementation(
        requested=args.attn_implementation,
        device=runtime_device,
        dtype=runtime_dtype,
    ) or "none"

    print("=" * 70)
    print("MOSS-TTS Unified Interface")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Attention: {args.attn_implementation}")
    print(f"Host: {args.host}:{args.port}")
    print(f"Share: {args.share}")
    print(f"Examples loaded: {len(EXAMPLE_ROWS)}")
    print("=" * 70)

    # Preload the main TTS model at startup (skip on HF Spaces where GPU time is metered)
    preload_enabled = parse_bool_env(PRELOAD_ENV_VAR, default=not bool(os.getenv("SPACE_ID")))
    if preload_enabled:
        preload_started_at = time.monotonic()
        print(
            f"[Startup] Preloading TTS backend: device={args.device}, attn={args.attn_implementation}",
            flush=True,
        )
        try:
            load_model("tts", args.device, args.attn_implementation)
            print(
                f"[Startup] Backend preload finished in {time.monotonic() - preload_started_at:.2f}s",
                flush=True,
            )
        except Exception as exc:
            print(f"[Startup] Preload failed (will load on first request): {exc}", flush=True)
    else:
        print(f"[Startup] Skipping preload (set {PRELOAD_ENV_VAR}=1 to enable).", flush=True)

    print("\n⏳ Building interface...")
    app = build_unified_interface(args)

    print("✅ Interface ready!")
    print(f"🌐 Access at: http://{args.host}:{args.port}")
    if args.share:
        print("🔗 Public link will be generated...")
    print()

    app.queue(max_size=20, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()
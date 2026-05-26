"""MOSS-TTS tab — voice cloning, continuation, and direct generation."""

import os
import time
import traceback
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch

from config import (
    MODE_CLONE,
    MODE_CONTINUE,
    MODE_CONTINUE_CLONE,
    TTS_LANGUAGE_AUTO,
    TTS_LANGUAGE_CHOICES,
    TTS_VARIANT_LOCAL,
    TTS_VARIANT_V15,
    resolve_tts_language,
)
from model_loader import (
    _truncate_reference_audio,
    download_model_files_for_keys,
    load_model,
)
from utils import (
    EXAMPLE_ROWS,
    build_tts_conversation,
    render_mode_hint,
    supports_duration_control,
    update_duration_controls,
)

# Insert [pause X.Ys] at the text cursor when possible; otherwise append (Gradio 6).
_INSERT_PAUSE_JS = """
(text, duration) => {
    const marker = `[pause ${Math.max(0.1, Math.min(30, Number(duration))).toFixed(1)}s]`;
    const root = document.getElementById("moss_tts_text");
    const el = root
        ? root.querySelector("textarea, input[type=text]")
        : document.querySelector("#moss_tts_text textarea");
    if (!el) {
        return (text || "") + marker;
    }
    const value = text ?? el.value ?? "";
    const start = typeof el.selectionStart === "number" ? el.selectionStart : value.length;
    const end = typeof el.selectionEnd === "number" ? el.selectionEnd : start;
    const updated = value.slice(0, start) + marker + value.slice(end);
    try {
        el.value = updated;
        const pos = start + marker.length;
        el.setSelectionRange(pos, pos);
        el.dispatchEvent(new Event("input", { bubbles: true }));
    } catch (e) {}
    return updated;
}
"""


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_tts_inference(
    text: str,
    reference_audio: Optional[str],
    mode_with_reference: str,
    duration_control_enabled: bool,
    duration_tokens: int,
    model_variant: str,
    language_choice: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    device: str,
    attn_implementation: str,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    started_at = time.monotonic()
    _ref_tmp = None
    try:
        if not text or not text.strip():
            return None, "❌ Error: Please enter text to synthesize"

        model_key = "tts_local" if model_variant == TTS_VARIANT_LOCAL else "tts"
        model, processor, dev, sample_rate = load_model(model_key, device, attn_implementation)

        language_tag = None
        if model_key == "tts":
            language_tag = resolve_tts_language(language_choice)

        resolved_ref = None
        if reference_audio:
            _ref_tmp = _truncate_reference_audio(reference_audio)
            resolved_ref = _ref_tmp

        duration_enabled = bool(
            duration_control_enabled and supports_duration_control(mode_with_reference)
        )
        expected_tokens = int(duration_tokens) if duration_enabled else None

        conversations, mode, mode_name = build_tts_conversation(
            text=text,
            reference_audio=resolved_ref,
            mode_with_reference=mode_with_reference,
            expected_tokens=expected_tokens,
            processor=processor,
            language=language_tag,
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
        audio_np = (
            audio.detach().float().cpu().numpy()
            if isinstance(audio, torch.Tensor)
            else np.asarray(audio, dtype=np.float32)
        )
        if audio_np.ndim > 1:
            audio_np = audio_np.reshape(-1)
        audio_np = audio_np.astype(np.float32, copy=False)
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_i16 = (audio_np * 32767.0).astype(np.int16)

        if _ref_tmp and _ref_tmp != reference_audio:
            try:
                os.unlink(_ref_tmp)
            except OSError:
                pass

        elapsed = time.monotonic() - started_at
        lang_note = language_tag if language_tag else "auto"
        status = (
            f"✅ Done | mode: {mode_name} | elapsed: {elapsed:.2f}s | "
            f"language={lang_note} | "
            f"max_new_tokens={max_new_tokens}, "
            f"expected_tokens={expected_tokens if expected_tokens is not None else 'off'}, "
            f"temperature={temperature:.2f}, top_p={top_p:.2f}, top_k={top_k}"
        )
        return (sample_rate, audio_i16), status

    except Exception as e:
        if _ref_tmp and _ref_tmp != reference_audio:
            try:
                os.unlink(_ref_tmp)
            except OSError:
                pass
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def _download_tts_model(_model_variant: str) -> str:
    """Prefetch both tab variants and the shared MOSS-Audio-Tokenizer (codec) repo."""
    try:
        return download_model_files_for_keys(["tts", "tts_local"])
    except Exception as e:
        return f"❌ Download failed: {e}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_tts_tab(args):
    with gr.Column():
        gr.Markdown("### 🎙️ MOSS-TTS-v1.5 - High-Quality Voice Cloning")
        gr.Markdown(
            "Generate speech with or without reference audio. "
            "v1.5 adds 31-language tags, stabler cloning, and inline pauses like `[pause 3.2s]`."
        )

        tts_model_variant = gr.Radio(
            choices=[TTS_VARIANT_V15, TTS_VARIANT_LOCAL],
            value=TTS_VARIANT_V15,
            label="Model Variant",
            info="v1.5 (8B): multilingual tags, pauses, long-form stability. Local (1.7B): highest SIM, lighter VRAM.",
        )

        with gr.Row():
            with gr.Column(scale=3):
                tts_language = gr.Dropdown(
                    choices=TTS_LANGUAGE_CHOICES,
                    value=TTS_LANGUAGE_AUTO,
                    label="Language tag (v1.5 only)",
                    info="Set when the text language is known (recommended for non Chinese/English).",
                    allow_custom_value=False,
                )
                tts_text = gr.Textbox(
                    label="Text to Synthesize",
                    lines=8,
                    elem_id="moss_tts_text",
                    placeholder=(
                        "Enter text here. Use [pause 3.2s] for explicit pauses (v1.5). "
                        "In Continuation modes, prepend the reference audio transcript."
                    ),
                )
                with gr.Row():
                    tts_pause_duration = gr.Slider(
                        minimum=0.1,
                        maximum=30.0,
                        value=3.2,
                        step=0.1,
                        label="Pause duration (seconds)",
                        scale=3,
                    )
                    tts_insert_pause_btn = gr.Button(
                        "Insert [pause …]",
                        scale=1,
                        size="sm",
                        variant="secondary",
                    )
                gr.Markdown(
                    "Click **Insert [pause …]** to add a marker at the cursor "
                    "(or at the end if the text box is not focused). v1.5 only."
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

                tts_download_btn = gr.Button("📥 Download Model", variant="secondary")
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

        def _language_control_visibility(variant: str):
            if variant == TTS_VARIANT_V15:
                return gr.update(interactive=True)
            return gr.update(interactive=False, value=TTS_LANGUAGE_AUTO)

        tts_model_variant.change(
            fn=_language_control_visibility,
            inputs=[tts_model_variant],
            outputs=[tts_language],
        )

        tts_insert_pause_btn.click(
            fn=None,
            inputs=[tts_text, tts_pause_duration],
            outputs=tts_text,
            js=_INSERT_PAUSE_JS,
        )

        # Mode hint reactivity
        for trigger in [tts_reference, tts_mode]:
            trigger.change(
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
                    return [gr.update()] * 7
                row_idx = (
                    int(evt.index[0]) if isinstance(evt.index, (tuple, list)) else int(evt.index)
                )
                if row_idx < 0 or row_idx >= len(EXAMPLE_ROWS):
                    return [gr.update()] * 7
                _, audio_path, example_text = EXAMPLE_ROWS[row_idx]
                dur_slider, dur_hint, dur_checkbox = update_duration_controls(
                    dur_enabled, example_text, dur_tokens, mode
                )
                return (
                    audio_path,
                    example_text,
                    TTS_LANGUAGE_AUTO,
                    render_mode_hint(audio_path, mode),
                    dur_slider,
                    dur_hint,
                    dur_checkbox,
                )

            tts_examples.select(
                fn=_apply_example,
                inputs=[tts_mode, tts_duration_enabled, tts_duration_tokens],
                outputs=[
                    tts_reference, tts_text, tts_language, tts_mode_hint,
                    tts_duration_tokens, tts_duration_hint, tts_duration_enabled,
                ],
            )

        tts_generate_btn.click(
            fn=lambda *x: run_tts_inference(*x, args.device, args.attn_implementation),
            inputs=[
                tts_text, tts_reference, tts_mode,
                tts_duration_enabled, tts_duration_tokens,
                tts_model_variant, tts_language,
                tts_temp, tts_top_p, tts_top_k, tts_rep_penalty, tts_max_tokens,
            ],
            outputs=[tts_output, tts_status],
        )

        tts_download_btn.click(
            fn=_download_tts_model,
            inputs=[tts_model_variant],
            outputs=[tts_status],
        )

"""MOSS-TTS tab — voice cloning, continuation, and direct generation."""

import os
import time
import traceback
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch

from config import MODE_CLONE, MODE_CONTINUE, MODE_CONTINUE_CLONE
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

        model_key = "tts_local" if model_variant == "MOSS-TTS-Local (1.7B)" else "tts"
        model, processor, dev, sample_rate = load_model(model_key, device, attn_implementation)

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
        gr.Markdown("### 🎙️ MOSS-TTS - High-Quality Voice Cloning")
        gr.Markdown(
            "Generate speech with or without reference audio. "
            "Supports voice cloning and continuation modes."
        )

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
                    return [gr.update()] * 6
                row_idx = (
                    int(evt.index[0]) if isinstance(evt.index, (tuple, list)) else int(evt.index)
                )
                if row_idx < 0 or row_idx >= len(EXAMPLE_ROWS):
                    return [gr.update()] * 6
                _, audio_path, example_text = EXAMPLE_ROWS[row_idx]
                dur_slider, dur_hint, dur_checkbox = update_duration_controls(
                    dur_enabled, example_text, dur_tokens, mode
                )
                return (
                    audio_path,
                    example_text,
                    render_mode_hint(audio_path, mode),
                    dur_slider,
                    dur_hint,
                    dur_checkbox,
                )

            tts_examples.select(
                fn=_apply_example,
                inputs=[tts_mode, tts_duration_enabled, tts_duration_tokens],
                outputs=[
                    tts_reference, tts_text, tts_mode_hint,
                    tts_duration_tokens, tts_duration_hint, tts_duration_enabled,
                ],
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

        tts_download_btn.click(
            fn=_download_tts_model,
            inputs=[tts_model_variant],
            outputs=[tts_status],
        )

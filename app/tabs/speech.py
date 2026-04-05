"""MOSS-Speech tab — true speech-to-speech language model without text guidance."""

import os
import time
import traceback
import uuid
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from transformers import (
    AutoModel,
    AutoProcessor,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)


# ---------------------------------------------------------------------------
# Stopping criteria (mirrors the official demo)
# ---------------------------------------------------------------------------

class _StopOnToken(StoppingCriteria):
    """Stop generation once the final token equals the provided stop ID."""

    def __init__(self, stop_id: int) -> None:
        super().__init__()
        self.stop_id = stop_id

    def __call__(self, input_ids: torch.LongTensor, scores) -> bool:
        return input_ids[0, -1].item() == self.stop_id


# ---------------------------------------------------------------------------
# Lazy model loader (separate from the TTS family loader because
# MOSS-Speech uses a different processor API and codec)
# ---------------------------------------------------------------------------

_speech_model = None
_speech_processor = None
_speech_device = None


def _load_speech_model(device_str: str):
    global _speech_model, _speech_processor, _speech_device
    if _speech_model is not None:
        return _speech_model, _speech_processor, _speech_device

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    from config import SPEECH_MODEL_PATH, SPEECH_CODEC_PATH
    model_path = SPEECH_MODEL_PATH
    codec_path = SPEECH_CODEC_PATH

    print(f"Loading MOSS-Speech from {model_path}…")

    # Resolve local path on Windows
    import sys
    if sys.platform == "win32" and "/" in model_path and not os.path.isdir(model_path):
        from huggingface_hub import snapshot_download
        local_model_path = snapshot_download(model_path)
    else:
        local_model_path = model_path

    if sys.platform == "win32" and "/" in codec_path and not os.path.isdir(codec_path):
        from huggingface_hub import snapshot_download
        local_codec_path = snapshot_download(codec_path)
    else:
        local_codec_path = codec_path

    processor = AutoProcessor.from_pretrained(
        local_model_path,
        codec_path=local_codec_path,
        device=str(device),
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        local_model_path, trust_remote_code=True,
    ).to(device).eval()

    _speech_model = model
    _speech_processor = processor
    _speech_device = device
    print("✓ MOSS-Speech loaded")
    return model, processor, device


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "speech_audio_cache")
os.makedirs(_AUDIO_DIR, exist_ok=True)

_DEFAULT_PROMPT_AUDIO = os.path.join(
    os.path.dirname(__file__), "..", "assets", "audio", "prompt_cn.wav"
)


def _save_audio_numpy(sr: int, arr: np.ndarray, prefix: str = "audio") -> str:
    if arr.ndim > 1:
        arr = arr[:, 0]
    path = os.path.join(_AUDIO_DIR, f"{prefix}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.wav")
    sf.write(path, arr, sr, format="WAV")
    return path


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_speech_inference(
    mode: str,
    audio_input,
    text_input: str,
    system_prompt: str,
    decoder_audio_prompt,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    min_new_tokens: int,
    device: str,
) -> Tuple[Optional[str], Optional[Tuple[int, np.ndarray]], str]:
    """Returns (text_output, audio_output_tuple, status_message)."""
    try:
        model, processor, dev = _load_speech_model(device)

        # Determine output modality from mode
        if mode.endswith("speech_response"):
            output_modality = "audio"
            sys_default = "You are a helpful voice assistant. Answer the user's questions with spoken responses."
        else:
            output_modality = "text"
            sys_default = "You are a helpful assistant. Answer the user's questions with text."

        system_prompt = (system_prompt or "").strip() or sys_default

        # Build conversation
        messages = []
        messages.append({"role": "system", "content": system_prompt})

        if mode.startswith("speech_instruct"):
            if audio_input is None:
                return None, None, "❌ Error: Speech input mode requires audio input"
            sr_in, arr_in = audio_input
            user_audio_path = _save_audio_numpy(sr_in, arr_in, prefix="user")
            messages.append({"role": "user", "content": {"path": user_audio_path, "type": "audio/wav"}})
        else:
            txt = (text_input or "").strip()
            if not txt:
                return None, None, "❌ Error: Text input mode requires text input"
            messages.append({"role": "user", "content": txt})

        # Decoder audio prompt
        decoder_path = None
        if decoder_audio_prompt is not None:
            sr_d, arr_d = decoder_audio_prompt
            decoder_path = _save_audio_numpy(sr_d, arr_d, prefix="decoder")
        elif os.path.exists(_DEFAULT_PROMPT_AUDIO):
            decoder_path = _DEFAULT_PROMPT_AUDIO

        # Stopping criteria
        tokenizer = processor.tokenizer
        stop_ids = [
            tokenizer.pad_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
        ]
        stopping_criteria = StoppingCriteriaList(
            [_StopOnToken(sid) for sid in stop_ids if sid is not None]
        )

        gen_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            use_cache=True,
        )

        encoded = processor([messages], [output_modality])
        with torch.inference_mode():
            token_ids = model.generate(
                input_ids=encoded["input_ids"].to(dev),
                attention_mask=encoded["attention_mask"].to(dev),
                generation_config=gen_config,
                stopping_criteria=stopping_criteria,
            )

        results = processor.decode(
            token_ids.to(dev),
            [output_modality],
            decoder_audio_prompt_path=decoder_path,
        )

        result_obj = results[0]

        if output_modality == "audio":
            if result_obj.audio is None:
                return None, None, "❌ Error: Model failed to generate speech"
            audio_np = result_obj.audio.squeeze(0).cpu().numpy()
            sr_out = result_obj.sampling_rate
            return None, (sr_out, audio_np), "✅ Speech-to-speech generation completed!"
        else:
            text_out = result_obj.generated_text
            if not text_out:
                return None, None, "❌ Error: Model failed to generate text"
            return text_out, None, "✅ Speech-to-text generation completed!"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, error_msg


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_speech_tab(args):
    with gr.Column():
        gr.Markdown("### 🗣️ MOSS-Speech — True Speech-to-Speech Language Model")
        gr.Markdown(
            "A bilingual (Chinese/English) native speech-to-speech model that works without text guidance. "
            "Supports speech-in/text-in → speech-out/text-out in all four combinations."
        )

        sp_mode = gr.Radio(
            choices=[
                ("Speech In → Speech Out", "speech_instruct_speech_response"),
                ("Speech In → Text Out", "speech_instruct_text_response"),
                ("Text In → Speech Out", "text_instruct_speech_response"),
                ("Text In → Text Out", "text_instruct_text_response"),
            ],
            value="speech_instruct_speech_response",
            label="Interaction Mode",
        )

        sp_system_prompt = gr.Textbox(
            label="System Prompt",
            value="You are a helpful voice assistant. Answer the user's questions with spoken responses.",
            lines=2,
        )

        with gr.Row():
            with gr.Column(scale=1):
                sp_audio_input = gr.Audio(
                    type="numpy", label="Speech Input", visible=True,
                )
                sp_text_input = gr.Textbox(
                    label="Text Input",
                    placeholder="Type your question here…",
                    lines=4,
                    visible=False,
                )
                sp_decoder_prompt = gr.Audio(
                    type="numpy",
                    label="Decoder Audio Prompt (Optional — controls output voice timbre)",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    sp_temp = gr.Slider(0.1, 2.0, value=0.6, step=0.1, label="Temperature")
                    sp_top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top P")
                    sp_top_k = gr.Slider(1, 100, value=20, step=1, label="Top K")
                    sp_rep_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.1, label="Repetition Penalty")
                    sp_max_tokens = gr.Slider(1, 2000, value=500, step=1, label="Max New Tokens")
                    sp_min_tokens = gr.Slider(0, 100, value=0, step=1, label="Min New Tokens")

                sp_generate_btn = gr.Button("🗣️ Generate", variant="primary", size="lg")

            with gr.Column(scale=1):
                sp_audio_output = gr.Audio(label="Speech Output", visible=True)
                sp_text_output = gr.Textbox(label="Text Output", lines=8, interactive=False, visible=False)
                sp_status = gr.Textbox(label="Status", lines=3, interactive=False)

                gr.Markdown("**About MOSS-Speech:**")
                gr.Markdown(
                    "- 9B parameter model\n"
                    "- True speech-to-speech without text guidance\n"
                    "- Preserves tone, emotion, and prosody\n"
                    "- Bilingual: Chinese and English\n"
                    "- Uses MOSS-Speech-Codec (0.9B) for audio encoding/decoding"
                )

        # Mode change: toggle input/output visibility and system prompt
        def _on_mode_change(mode_val):
            is_speech_in = mode_val.startswith("speech_instruct")
            is_speech_out = mode_val.endswith("speech_response")
            if is_speech_out:
                sys_p = "You are a helpful voice assistant. Answer the user's questions with spoken responses."
            else:
                sys_p = "You are a helpful assistant. Answer the user's questions with text."
            return (
                gr.update(visible=is_speech_in),
                gr.update(visible=not is_speech_in),
                gr.update(visible=is_speech_out),
                gr.update(visible=not is_speech_out),
                sys_p,
            )

        sp_mode.change(
            fn=_on_mode_change,
            inputs=[sp_mode],
            outputs=[sp_audio_input, sp_text_input, sp_audio_output, sp_text_output, sp_system_prompt],
        )

        sp_generate_btn.click(
            fn=lambda *x: run_speech_inference(*x, args.device),
            inputs=[
                sp_mode, sp_audio_input, sp_text_input, sp_system_prompt,
                sp_decoder_prompt, sp_temp, sp_top_p, sp_top_k,
                sp_rep_penalty, sp_max_tokens, sp_min_tokens,
            ],
            outputs=[sp_text_output, sp_audio_output, sp_status],
        )

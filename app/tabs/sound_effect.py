"""MOSS-SoundEffect tab — generate environmental sounds from text."""

import traceback
from typing import Optional, Tuple

import gradio as gr
import numpy as np

from config import TOKENS_PER_SECOND
from core import Sampling, generate_and_decode, load_model
from core.download import download_model_files_for_keys


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

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
    try:
        if not description or not description.strip():
            return None, "❌ Error: Please enter sound description"

        model, processor, dev, sample_rate = load_model("sound_effect", device, attn_implementation)

        expected_tokens = max(1, int(duration_seconds * TOKENS_PER_SECOND))
        conversation = [processor.build_user_message(ambient_sound=description, tokens=expected_tokens)]

        # Cap generation to the requested duration (+25% slack) so we never grow
        # the KV cache far beyond what the clip needs. A runaway max_new_tokens
        # bloats VRAM and stalls generation long after the audio is complete.
        effective_max_tokens = min(int(max_new_tokens), int(expected_tokens * 1.25) + 16)

        sampling = Sampling(temperature, top_p, top_k, repetition_penalty)
        audio_i16 = generate_and_decode(
            model, processor, dev, conversation, "generation", sampling, effective_max_tokens
        )
        return (sample_rate, audio_i16), "✅ Sound effect generated!"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def _download_sound_effect_models() -> str:
    """Prefetch main checkpoint and MOSS-Audio-Tokenizer (codec / tokenizer weights)."""
    try:
        return download_model_files_for_keys(["sound_effect"])
    except Exception as e:
        return f"❌ Download failed: {e}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_sound_effect_tab(args):
    with gr.Column():
        gr.Markdown("### 🔊 MOSS-SoundEffect - Generate Sound Effects")
        gr.Markdown("Create sound effects and environmental audio from text descriptions.")

        with gr.Row():
            with gr.Column(scale=1):
                se_description = gr.Textbox(
                    label="Sound Description",
                    lines=4,
                    placeholder="Describe the sound you want (e.g., 'Thunder and rain', 'City traffic', 'Forest birds')…",
                )
                se_duration = gr.Slider(
                    1, 60, value=10, step=1,
                    label="Duration (seconds)",
                    info="Target length of the generated sound effect.",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    se_temp = gr.Slider(0.1, 3.0, value=1.5, step=0.05, label="Temperature")
                    se_top_p = gr.Slider(0.1, 1.0, value=0.6, step=0.01, label="Top P")
                    se_top_k = gr.Slider(1, 200, value=50, step=1, label="Top K")
                    se_rep_penalty = gr.Slider(0.8, 2.0, value=1.2, step=0.05, label="Repetition Penalty")
                    se_max_tokens = gr.Slider(
                        256, 8192, value=2048, step=128, label="Max New Tokens",
                        info="Hard cap; generation is also auto-limited to the chosen duration.",
                    )

                se_download_btn = gr.Button("📥 Download Model", variant="secondary")
                se_generate_btn = gr.Button("🎵 Generate Sound", variant="primary", size="lg")

            with gr.Column(scale=1):
                se_output = gr.Audio(label="Generated Sound")
                se_status = gr.Textbox(label="Status", lines=3, interactive=False)

                gr.Markdown("**Example Sounds:**")
                gr.Markdown(
                    "- Ocean waves crashing on the beach\n"
                    "- Busy city street with traffic\n"
                    "- Birds chirping in a forest\n"
                    "- Thunderstorm with heavy rain"
                )

        se_generate_btn.click(
            fn=lambda *x: run_sound_effect_inference(*x, args.device, args.attn_implementation),
            inputs=[se_description, se_duration, se_temp, se_top_p, se_top_k, se_rep_penalty, se_max_tokens],
            outputs=[se_output, se_status],
        )

        se_download_btn.click(
            fn=_download_sound_effect_models,
            inputs=[],
            outputs=[se_status],
        )

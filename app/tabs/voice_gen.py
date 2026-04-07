"""MOSS-VoiceGenerator tab — design voices from text descriptions."""

import traceback
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch

from model_loader import download_model_files_for_keys, load_model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

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
    try:
        if not instruction or not instruction.strip():
            return None, "❌ Error: Please enter voice description"
        if not text or not text.strip():
            return None, "❌ Error: Please enter text to synthesize"

        model, processor, dev, sample_rate = load_model("voice_gen", device, attn_implementation)

        conversation = [processor.build_user_message(instruction=instruction, text=text)]
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
                audio_repetition_penalty=repetition_penalty,
            )

        messages = processor.decode(outputs)
        if messages and len(messages) > 0:
            audio_np = messages[0].audio_codes_list[0].cpu().numpy()
            return (sample_rate, audio_np), "✅ Voice generation completed!"

        return None, "❌ Error: No audio generated"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def _download_voice_gen_models() -> str:
    """Prefetch main checkpoint and MOSS-Audio-Tokenizer (codec / tokenizer weights)."""
    try:
        return download_model_files_for_keys(["voice_gen"])
    except Exception as e:
        return f"❌ Download failed: {e}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_voice_gen_tab(args):
    with gr.Column():
        gr.Markdown("### 🎨 MOSS-VoiceGenerator - Design Voices from Text")
        gr.Markdown("Create unique voices by describing them in natural language.")

        with gr.Row():
            with gr.Column(scale=1):
                vg_instruction = gr.Textbox(
                    label="Voice Description",
                    lines=4,
                    placeholder="Describe the voice you want (e.g., 'A young female with a cheerful and energetic tone')…",
                )
                vg_text = gr.Textbox(
                    label="Text to Synthesize",
                    lines=6,
                    placeholder="Enter the text for the voice to speak…",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    vg_temp = gr.Slider(0.1, 3.0, value=1.5, step=0.05, label="Temperature")
                    vg_top_p = gr.Slider(0.1, 1.0, value=0.6, step=0.01, label="Top P")
                    vg_top_k = gr.Slider(1, 200, value=50, step=1, label="Top K")
                    vg_rep_penalty = gr.Slider(0.8, 2.0, value=1.1, step=0.05, label="Repetition Penalty")
                    vg_max_tokens = gr.Slider(256, 8192, value=4096, step=128, label="Max New Tokens")

                vg_download_btn = gr.Button("📥 Download Model", variant="secondary")
                vg_generate_btn = gr.Button("✨ Generate Voice", variant="primary", size="lg")

            with gr.Column(scale=1):
                vg_output = gr.Audio(label="Generated Audio")
                vg_status = gr.Textbox(label="Status", lines=3, interactive=False)

                gr.Markdown("**Example Descriptions:**")
                gr.Markdown(
                    "- A middle-aged male with a deep, authoritative voice\n"
                    "- A young child with a playful tone\n"
                    "- An elderly woman with a warm, gentle voice"
                )

        vg_generate_btn.click(
            fn=lambda *x: run_voice_gen_inference(*x, args.device, args.attn_implementation),
            inputs=[vg_instruction, vg_text, vg_temp, vg_top_p, vg_top_k, vg_rep_penalty, vg_max_tokens],
            outputs=[vg_output, vg_status],
        )

        vg_download_btn.click(
            fn=_download_voice_gen_models,
            inputs=[],
            outputs=[vg_status],
        )

"""MOSS-TTS-Realtime tab — low-latency streaming TTS for voice agents."""

import traceback
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch

from model_loader import load_realtime_model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

CODEC_SAMPLE_RATE = 24000


def run_realtime_inference(
    text: str,
    reference_audio: Optional[str],
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    repetition_window: int,
    max_length: int,
    device: str,
    attn_implementation: str,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    try:
        if not text or not text.strip():
            return None, "❌ Error: Please enter text to synthesize"

        inferencer, codec, dev, sample_rate = load_realtime_model(device, attn_implementation)

        text_list = [text]
        ref_list = [reference_audio if reference_audio else ""]

        result = inferencer.generate(
            text=text_list,
            reference_audio_path=ref_list,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            repetition_window=repetition_window,
            device=dev,
        )

        wav_chunks = []
        for generated_tokens in result:
            output = torch.tensor(generated_tokens).to(dev)
            decode_result = codec.decode(output.permute(1, 0), chunk_duration=8)
            wav = decode_result["audio"][0].cpu().detach()
            wav_chunks.append(wav)

        if not wav_chunks:
            return None, "❌ Error: No audio generated"

        audio = torch.cat(wav_chunks, dim=-1)
        audio_np = audio.squeeze().numpy()
        return (sample_rate, audio_np), "✅ Realtime generation completed!"

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_realtime_tab(args):
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
                    placeholder="Enter text here…",
                )
                rt_reference = gr.Audio(
                    label="Reference Audio (Optional)",
                    type="filepath",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    rt_temp = gr.Slider(0.1, 3.0, value=0.8, step=0.05, label="Temperature")
                    rt_top_p = gr.Slider(0.1, 1.0, value=0.6, step=0.01, label="Top P")
                    rt_top_k = gr.Slider(1, 200, value=30, step=1, label="Top K")
                    rt_rep_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition Penalty")
                    rt_rep_window = gr.Slider(10, 200, value=50, step=5, label="Repetition Window")
                    rt_max_length = gr.Slider(1000, 10000, value=5000, step=500, label="Max Length")

                rt_generate_btn = gr.Button("⚡ Generate (Realtime)", variant="primary", size="lg")

            with gr.Column(scale=1):
                rt_output = gr.Audio(label="Generated Audio")
                rt_status = gr.Textbox(label="Status", lines=3, interactive=False)

                gr.Markdown("**About this model:**")
                gr.Markdown(
                    "- Architecture: MossTTSRealtime (1.7B)\n"
                    "- TTFB: ~180 ms (after warm-up)\n"
                    "- Ideal for voice agents paired with LLMs\n"
                    "- Supports multi-turn context via KV cache reuse\n"
                    "- Uses MOSS-Audio-Tokenizer codec"
                )

        rt_generate_btn.click(
            fn=lambda *x: run_realtime_inference(*x, args.device, args.attn_implementation),
            inputs=[rt_text, rt_reference, rt_temp, rt_top_p, rt_top_k, rt_rep_penalty, rt_rep_window, rt_max_length],
            outputs=[rt_output, rt_status],
        )

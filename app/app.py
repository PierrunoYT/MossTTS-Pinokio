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
import os
import sys
import time

import gradio as gr
import torch

from config import MODELS, PRELOAD_ENV_VAR
from model_loader import load_model, resolve_attn_implementation
from utils import EXAMPLE_ROWS, parse_bool_env, parse_port
from tabs.tts import build_tts_tab
from tabs.ttsd import build_ttsd_tab
from tabs.voice_gen import build_voice_gen_tab
from tabs.sound_effect import build_sound_effect_tab
from tabs.realtime import build_realtime_tab
from tabs.speech import build_speech_tab
from tabs.info import build_info_tab


# ---------------------------------------------------------------------------
# Interface builder
# ---------------------------------------------------------------------------

_CSS = """
.app-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
.app-header h1 { margin: 0; font-size: 2.5em; }
.app-header p  { margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9; }
"""


def build_unified_interface(args):
    with gr.Blocks(title="MOSS-TTS Unified Interface", css=_CSS, theme=gr.themes.Soft()) as app:
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
            with gr.Tab("🗣️ Speech-to-Speech"):
                build_speech_tab(args)
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MOSS-TTS Unified Interface")
    parser.add_argument("--model_path", type=str, default=MODELS["tts"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn_implementation", type=str, default="auto")
    _default_host = "127.0.0.1" if sys.platform == "win32" else "0.0.0.0"
    parser.add_argument("--host", type=str, default=_default_host)
    parser.add_argument(
        "--port",
        type=int,
        default=parse_port(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT")), 7860),
    )
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    args.host = os.getenv("GRADIO_SERVER_NAME", args.host)
    args.port = parse_port(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT")), args.port)

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
    print(f"Device:     {args.device}")
    print(f"Attention:  {args.attn_implementation}")
    print(f"Host:       {args.host}:{args.port}")
    print(f"Share:      {args.share}")
    print(f"Examples:   {len(EXAMPLE_ROWS)} loaded")
    print("=" * 70)

    preload_enabled = parse_bool_env(PRELOAD_ENV_VAR, default=not bool(os.getenv("SPACE_ID")))
    if preload_enabled:
        t0 = time.monotonic()
        print(f"[Startup] Preloading TTS: device={args.device}, attn={args.attn_implementation}")
        try:
            load_model("tts", args.device, args.attn_implementation)
            print(f"[Startup] Preload done in {time.monotonic() - t0:.2f}s")
        except Exception as exc:
            print(f"[Startup] Preload failed (will load on first request): {exc}")
    else:
        print(f"[Startup] Skipping preload (set {PRELOAD_ENV_VAR}=1 to enable).")

    print("\n⏳ Building interface…")
    app = build_unified_interface(args)
    print("✅ Interface ready!")
    print(f"🌐 http://{args.host}:{args.port}\n")

    app.queue(max_size=20, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()

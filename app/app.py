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

# Apply transformers remote-code shims and Windows asyncio noise suppression
# before anything touches the model stack.
from core.compat import apply_compat_shims

apply_compat_shims()

from config import (
    DEFAULT_MODEL_CACHE_SIZE,
    MODEL_CACHE_SIZE_ENV_VAR,
    MODELS,
    PRELOAD_ENV_VAR,
    QUANTIZATION_ENV_VAR,
)
from core import RuntimeConfig, load_model, resolve_quantization, set_quantization_override
from core.quant import bitsandbytes_available
from utils import EXAMPLE_ROWS, parse_bool_env, parse_port
from tabs.tts import build_tts_tab
from tabs.ttsd import build_ttsd_tab
from tabs.voice_gen import build_voice_gen_tab
from tabs.sound_effect import build_sound_effect_tab
from tabs.realtime import build_realtime_tab
from tabs.nano import build_nano_tab
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


_QUANT_LABELS = {
    "auto": "Auto (recommended)",
    "8bit": "8-bit (higher quality, ~2× the VRAM of 4-bit)",
    "4bit": "4-bit (smallest VRAM, slight quality loss)",
    "none": "None / bf16 (best quality, needs most VRAM)",
}
_QUANT_FROM_LABEL = {v: k for k, v in _QUANT_LABELS.items()}


def _build_quantization_control():
    """A header dropdown to switch weight quantization at runtime.

    Only meaningful on CUDA with bitsandbytes installed; the new mode applies on
    the next model load (the resolved mode is part of the model cache key, so a
    cached model in another mode is reloaded). Hidden on CPU-only setups.
    """
    if not (torch.cuda.is_available() and bitsandbytes_available()):
        return

    current = resolve_quantization()
    # Reflect the configured starting mode; prefer showing "auto" if that's what
    # the env/CLI requested rather than its resolved concrete value.
    requested = (os.getenv(QUANTIZATION_ENV_VAR) or "none").strip().lower()
    start = requested if requested in _QUANT_LABELS else current
    if start not in _QUANT_LABELS:
        start = "none"

    with gr.Row():
        dropdown = gr.Dropdown(
            choices=list(_QUANT_LABELS.values()),
            value=_QUANT_LABELS[start],
            label="⚙️ Quantization (weight precision)",
            info="Applies on the next model load. 4-bit fits 8B models on ~6GB; "
            "8-bit trades more VRAM for better fidelity.",
            scale=2,
        )
        status = gr.Markdown("")

    def _on_change(label: str) -> str:
        mode = _QUANT_FROM_LABEL.get(label, "auto")
        effective = set_quantization_override(mode)
        return f"✓ Set to **{mode}** (effective: `{effective}`). Reloads on next generate."

    dropdown.change(_on_change, inputs=[dropdown], outputs=[status])


def build_unified_interface(args):
    with gr.Blocks(title="MOSS-TTS Unified Interface") as app:
        gr.HTML("""
        <div class="app-header">
            <h1>🎵 MOSS-TTS Family</h1>
            <p>Unified Interface for All Models</p>
        </div>
        """)

        _build_quantization_control()

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
            with gr.Tab("🧩 Nano (ONNX CPU)"):
                build_nano_tab(args)
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
    _default_host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    parser.add_argument("--host", type=str, default=_default_host)
    parser.add_argument(
        "--port",
        type=int,
        default=parse_port(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT")), 7860),
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["auto", "none", "8bit", "4bit"],
        help="Weight quantization (requires bitsandbytes + CUDA). Shrinks VRAM "
        "use so 8B models fit on smaller GPUs. 'auto' picks 4-bit on cards "
        "<=32GB when bitsandbytes is present, else bf16. Default: none (env "
        f"{QUANTIZATION_ENV_VAR} also honoured).",
    )
    parser.add_argument(
        "--model_cache_size",
        type=int,
        default=None,
        help="Max models kept resident in GPU memory at once (default "
        f"{DEFAULT_MODEL_CACHE_SIZE}; env {MODEL_CACHE_SIZE_ENV_VAR}). "
        "Raise on large-VRAM cards to avoid reload latency.",
    )
    args = parser.parse_args()

    # CLI flags win, but fall back to env vars so Pinokio/Spaces configs work too.
    if args.quantization is not None:
        os.environ[QUANTIZATION_ENV_VAR] = args.quantization
    if args.model_cache_size is not None:
        os.environ[MODEL_CACHE_SIZE_ENV_VAR] = str(max(1, args.model_cache_size))

    MODELS["tts"] = args.model_path
    runtime = RuntimeConfig(args.device, args.attn_implementation)
    args.device = str(runtime.device)
    args.attn_implementation = runtime.resolved_attn() or "none"

    print("=" * 70)
    print("MOSS-TTS Unified Interface")
    print("=" * 70)
    print(f"Device:     {args.device}")
    print(f"Attention:  {args.attn_implementation}")
    print(f"Quantize:   {os.getenv(QUANTIZATION_ENV_VAR, 'none')}")
    print(f"GPU cache:  {os.getenv(MODEL_CACHE_SIZE_ENV_VAR, str(DEFAULT_MODEL_CACHE_SIZE))} model(s) resident")
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

    app.queue(max_size=20, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        ssr_mode=False,
        css=_CSS,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()

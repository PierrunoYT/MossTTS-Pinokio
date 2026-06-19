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
import asyncio
import os
import sys
import time

import gradio as gr
import torch

# ---------------------------------------------------------------------------
# Transformers compat: MOSS remote code (trust_remote_code=True) references
# mapping constants such as `MODALITY_TO_BASE_CLASS_MAPPING` /
# `MODALITY_TO_AE_CLASS_MAPPING`, which have been renamed across transformers
# versions (the current name is `AUTO_TO_BASE_CLASS_MAPPING`). Resolve the
# mapping from whichever name exists and re-expose it under every alias the
# remote code may look up so its import doesn't crash. This shim is a no-op
# when the expected attributes already exist.
# ---------------------------------------------------------------------------
try:
    import transformers as _tf
    import transformers.processing_utils as _pu

    # The MOSS remote code references several aliases of the same mapping
    # that have been renamed across transformers versions.  Resolve the
    # mapping from whichever name exists, then expose it under every name
    # the remote code may look up, on both the top-level package and the
    # processing_utils / auto sub-modules.
    _alias_names = (
        "MODALITY_TO_BASE_CLASS_MAPPING",
        "MODALITY_TO_AE_CLASS_MAPPING",
        "AUTO_TO_BASE_CLASS_MAPPING",
    )

    def _resolve_mapping():
        for _mod in (_pu, _tf):
            for _name in _alias_names:
                _candidate = getattr(_mod, _name, None)
                if _candidate is not None:
                    return _candidate
        return None

    _mapping = _resolve_mapping()
    if _mapping is not None:
        _targets = [_tf, _pu]
        try:
            import transformers.models.auto as _auto  # noqa: PLC0415

            _targets.append(_auto)
        except Exception:
            pass
        for _mod in _targets:
            for _name in _alias_names:
                if not hasattr(_mod, _name):
                    setattr(_mod, _name, _mapping)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Transformers compat: MOSS remote code imports `PreTrainedConfig` from
# `transformers.configuration_utils`, but older transformers releases only
# expose it as `PretrainedConfig`. Alias the new name to the old class (and
# vice versa) so the remote code's import succeeds. No-op when both exist.
# ---------------------------------------------------------------------------
try:
    import transformers as _tf
    import transformers.configuration_utils as _cu

    _new_name, _old_name = "PreTrainedConfig", "PretrainedConfig"
    _cfg = getattr(_cu, _new_name, None) or getattr(_cu, _old_name, None)
    if _cfg is not None:
        for _mod in (_cu, _tf):
            for _name in (_new_name, _old_name):
                if not hasattr(_mod, _name):
                    setattr(_mod, _name, _cfg)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Windows: suppress the harmless "WinError 10054 - An existing connection was
# forcibly closed by the remote host" noise that asyncio's ProactorEventLoop
# raises whenever a browser tab closes mid-stream.  The error is benign but
# it pollutes logs and can cause unhandled-exception warnings on Python 3.9+.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        from asyncio import proactor_events as _pe

        _orig_call_connection_lost = _pe._ProactorBasePipeTransport._call_connection_lost  # type: ignore[attr-defined]

        def _quiet_call_connection_lost(self, exc):  # type: ignore[override]
            try:
                _orig_call_connection_lost(self, exc)
            except OSError:
                pass

        _pe._ProactorBasePipeTransport._call_connection_lost = _quiet_call_connection_lost  # type: ignore[attr-defined]
    except Exception:
        pass

from config import (
    DEFAULT_MODEL_CACHE_SIZE,
    MODEL_CACHE_SIZE_ENV_VAR,
    MODELS,
    PRELOAD_ENV_VAR,
    QUANTIZATION_ENV_VAR,
)
from model_loader import load_model, resolve_attn_implementation
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
    _default_host = "127.0.0.1" if sys.platform == "win32" else "0.0.0.0"
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
    print(f"🌐 http://{args.host}:{args.port}\n")

    app.queue(max_size=20, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        ssr_mode=False,
    )


if __name__ == "__main__":
    main()

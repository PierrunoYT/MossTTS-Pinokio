"""MOSS-TTS-Nano tab — lightweight ONNX CPU voice cloning."""

import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf

try:
    from wetext import Normalizer
except Exception:
    Normalizer = None


NANO_DIR = Path(__file__).resolve().parents[1] / "MOSS-TTS-Nano"
NANO_INFER_SCRIPT = NANO_DIR / "infer_onnx.py"


def _ensure_nano_ready() -> Optional[str]:
    if not NANO_DIR.exists():
        return "❌ MOSS-TTS-Nano repo not found at app/MOSS-TTS-Nano. Run install/update first."
    if not NANO_INFER_SCRIPT.exists():
        return "❌ Missing infer_onnx.py in app/MOSS-TTS-Nano."
    return None


def run_nano_inference(
    text: str,
    reference_audio: Optional[str],
    voice_preset: str,
    sample_mode: str,
    max_new_frames: int,
    cpu_threads: int,
    use_wetext_normalizer: bool,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    try:
        readiness_error = _ensure_nano_ready()
        if readiness_error:
            return None, readiness_error
        normalizer_method = "none"
        if use_wetext_normalizer:
            normalized_text, normalizer_method = _normalize_with_wetext_fallback(text)
        else:
            normalized_text = text
        if not normalized_text or not normalized_text.strip():
            return None, "❌ Error: Please enter text to synthesize."

        with tempfile.TemporaryDirectory(prefix="moss_nano_") as tmpdir:
            out_wav = Path(tmpdir) / "nano_output.wav"
            cmd = [
                sys.executable,
                "infer_onnx.py",
                "--text",
                normalized_text.strip(),
                "--output-audio-path",
                str(out_wav),
                "--sample-mode",
                sample_mode,
                "--max-new-frames",
                str(int(max_new_frames)),
                "--cpu-threads",
                str(int(cpu_threads)),
                "--disable-wetext-processing",
            ]
            if reference_audio:
                cmd.extend(["--prompt-audio-path", reference_audio])
            else:
                cmd.extend(["--voice", voice_preset])

            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"

            result = subprocess.run(
                cmd,
                cwd=str(NANO_DIR),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                stdout = (result.stdout or "").strip()
                details = stderr if stderr else stdout
                return None, f"❌ Nano inference failed (exit={result.returncode}).\n\n{details}"

            if not out_wav.exists():
                return None, "❌ Nano inference finished but no audio file was produced."

            audio_np, sample_rate = sf.read(str(out_wav), dtype="int16")
            if audio_np.ndim > 1:
                audio_np = audio_np[:, 0]
            status = "✅ MOSS-TTS-Nano ONNX generation completed!"
            if normalizer_method != "none":
                status += f" Text normalization: {normalizer_method}."
            return (int(sample_rate), audio_np), status

    except Exception:
        return None, f"❌ Error:\n{traceback.format_exc()}"


def update_nano_mode_hint(reference_audio: Optional[str]) -> str:
    if reference_audio:
        return "Mode: **Voice Clone** (using uploaded reference audio)"
    return "Mode: **Preset Voice** (no reference audio uploaded)"


def _normalize_web_text_basic(text: str) -> str:
    # Lightweight fallback when wetext is unavailable.
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("’", "'")
    return cleaned.strip()


def _normalize_with_wetext_fallback(text: str) -> Tuple[str, str]:
    raw = (text or "").strip()
    if not raw:
        return raw, "none"

    if Normalizer is not None:
        try:
            normalizer = Normalizer(lang="zh", operator="tn")
            normalized = normalizer.normalize(raw)
            if normalized and normalized.strip():
                return normalized.strip(), "wetext(zh,tn)"
        except Exception:
            pass
    return _normalize_web_text_basic(raw), "basic-fallback"


def build_nano_tab(_args):
    with gr.Column():
        gr.Markdown("### 🧩 MOSS-TTS-Nano (ONNX CPU)")
        gr.Markdown(
            "Integrated upstream `OpenMOSS/MOSS-TTS-Nano` in `app/MOSS-TTS-Nano` "
            "and runs `infer_onnx.py` for lightweight CPU-friendly synthesis."
        )

        with gr.Row():
            with gr.Column(scale=3):
                nano_text = gr.Textbox(
                    label="Text to Synthesize",
                    lines=6,
                    placeholder="Enter text here...",
                )
                nano_reference = gr.Audio(
                    label="Reference Audio (Optional, for voice cloning)",
                    type="filepath",
                )
                nano_mode_hint = gr.Markdown(update_nano_mode_hint(None))
                nano_voice = gr.Textbox(
                    label="Voice Preset (used if no reference audio)",
                    value="Junhao",
                )
                with gr.Accordion("Advanced ONNX Settings", open=False):
                    nano_normalize = gr.Checkbox(
                        value=True,
                        label="Use wetext normalization (fallback to basic)",
                    )
                    nano_sample_mode = gr.Radio(
                        choices=["fixed", "greedy", "full"],
                        value="fixed",
                        label="Sample Mode",
                    )
                    nano_max_frames = gr.Slider(
                        minimum=64, maximum=1200, value=375, step=1, label="Max New Frames"
                    )
                    nano_threads = gr.Slider(
                        minimum=1, maximum=16, value=4, step=1, label="CPU Threads"
                    )
                nano_generate_btn = gr.Button("🧩 Generate with Nano", variant="primary", size="lg")

            with gr.Column(scale=2):
                nano_output = gr.Audio(label="Generated Audio")
                nano_status = gr.Textbox(label="Status", lines=14, interactive=False)

        nano_reference.change(
            fn=update_nano_mode_hint,
            inputs=[nano_reference],
            outputs=[nano_mode_hint],
        )

        nano_generate_btn.click(
            fn=run_nano_inference,
            inputs=[
                nano_text,
                nano_reference,
                nano_voice,
                nano_sample_mode,
                nano_max_frames,
                nano_threads,
                nano_normalize,
            ],
            outputs=[nano_output, nano_status],
        )

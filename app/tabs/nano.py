"""MOSS-TTS-Nano tab — HF Space-style voice cloning integration."""

import logging
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    from wetext import Normalizer
except Exception:
    Normalizer = None


logger = logging.getLogger(__name__)
MODEL_ID = "OpenMOSS-Team/MOSS-TTS-Nano-100M"
AUDIO_TOKENIZER_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
NANO_DIR = Path(__file__).resolve().parents[1] / "MOSS-TTS-Nano"
OUTPUT_DIR = NANO_DIR / "generated_audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXAMPLE_TEXTS = {
    "English": (
        "The biggest lesson that can be read from 70 years of AI research is that general methods "
        "that leverage computation are ultimately the most effective, and by a large margin."
    ),
    "Chinese": "欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。今天我们将为您带来最新的人工智能研究进展。",
    "Japanese": "本日はNHKニュースをご覧いただきありがとうございます。最新のニュースをお伝えします。",
}

SAMPLE_AUDIO = {
    "English": str(NANO_DIR / "assets" / "audio" / "en_2.wav"),
    "Chinese": str(NANO_DIR / "assets" / "audio" / "zh_1.wav"),
    "Japanese": str(NANO_DIR / "assets" / "audio" / "jp_2.wav"),
}

_NANO_RUNTIME = None


def _ensure_nano_repo() -> Optional[str]:
    if not NANO_DIR.exists():
        return "❌ MOSS-TTS-Nano repo not found at app/MOSS-TTS-Nano. Run install/update first."
    return None


def _normalize_web_text_basic(text: str) -> str:
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("’", "'")
    return cleaned.strip()


def _normalize_with_wetext_fallback(text: str, lang: str) -> Tuple[str, str]:
    raw = (text or "").strip()
    if not raw:
        return raw, "none"
    lang_code = {"English": "en", "Chinese": "zh", "Japanese": "ja"}.get(lang, "zh")
    if Normalizer is not None:
        try:
            normalizer = Normalizer(lang=lang_code, operator="tn")
            normalized = normalizer.normalize(raw)
            if normalized and normalized.strip():
                return normalized.strip(), f"wetext({lang_code},tn)"
        except Exception:
            pass
    return _normalize_web_text_basic(raw), "basic-fallback"


def _load_nano_runtime(device_hint: str):
    global _NANO_RUNTIME
    if _NANO_RUNTIME is not None:
        return _NANO_RUNTIME

    repo_error = _ensure_nano_repo()
    if repo_error:
        raise RuntimeError(repo_error)

    runtime_device = "cuda" if (torch.cuda.is_available() and "cuda" in str(device_hint)) else "cpu"
    dtype = torch.bfloat16 if runtime_device == "cuda" else torch.float32

    logger.info("Loading Nano TTS model on %s", runtime_device)
    tts_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    tts_model.eval()
    if hasattr(tts_model, "_set_attention_implementation"):
        tts_model._set_attention_implementation("sdpa")

    logger.info("Loading Nano audio tokenizer")
    audio_tokenizer = AutoModel.from_pretrained(
        AUDIO_TOKENIZER_ID,
        trust_remote_code=True,
    )
    audio_tokenizer.eval()

    logger.info("Loading Nano text tokenizer")
    text_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    tts_model.to(runtime_device)
    audio_tokenizer.to(runtime_device)

    _NANO_RUNTIME = {
        "tts_model": tts_model,
        "audio_tokenizer": audio_tokenizer,
        "text_tokenizer": text_tokenizer,
        "device": runtime_device,
    }
    return _NANO_RUNTIME


def _safe_ref_path(example_lang: str, uploaded: Optional[str]) -> Optional[str]:
    if uploaded:
        return uploaded
    sample = SAMPLE_AUDIO.get(example_lang)
    if sample and Path(sample).exists():
        return sample
    return None


def on_example_select(lang: str) -> Tuple[str, Optional[str]]:
    return EXAMPLE_TEXTS.get(lang, ""), _safe_ref_path(lang, None)


def run_nano_inference(
    text: str,
    reference_audio: Optional[str],
    max_new_frames: int,
    do_sample: bool,
    seed: int,
    use_wetext_normalizer: bool,
    example_lang: str,
    device_hint: str,
) -> Tuple[Optional[str], str]:
    try:
        if not text or not text.strip():
            return None, "❌ Please enter text to synthesize."

        ref_audio = _safe_ref_path(example_lang, reference_audio)
        if not ref_audio:
            return None, "❌ Please upload reference audio (or pick a language sample)."

        normalizer_method = "none"
        normalized_text = text
        if use_wetext_normalizer:
            normalized_text, normalizer_method = _normalize_with_wetext_fallback(text, example_lang)

        try:
            runtime = _load_nano_runtime(device_hint)
            tts_model = runtime["tts_model"]
            audio_tokenizer = runtime["audio_tokenizer"]
            text_tokenizer = runtime["text_tokenizer"]
            device = runtime["device"]

            seed_int = int(seed) if seed is not None else 0
            if seed_int != 0:
                torch.manual_seed(seed_int)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed_int)

            out_path = OUTPUT_DIR / f"nano_{os.getpid()}_{abs(hash(normalized_text)) % 1000000}.wav"
            result = tts_model.inference(
                text=normalized_text,
                output_audio_path=str(out_path),
                mode="voice_clone",
                prompt_audio_path=ref_audio,
                text_tokenizer=text_tokenizer,
                audio_tokenizer=audio_tokenizer,
                audio_tokenizer_type="moss-audio-tokenizer-nano",
                device=device,
                max_new_frames=int(max_new_frames),
                do_sample=bool(do_sample),
                use_kv_cache=True,
                voice_clone_max_text_tokens=75,
            )

            audio_path = str(result.get("audio_path", out_path))
            if not Path(audio_path).exists():
                return None, "❌ Nano generation finished but produced no output file."

            status = f"✅ Nano generation completed on {device}."
            if normalizer_method != "none":
                status += f" Text normalization: {normalizer_method}."
            return audio_path, status
        except Exception as native_exc:
            logger.warning("Native Nano runtime failed; falling back to ONNX CLI: %s", native_exc)
            return _run_nano_onnx_fallback(
                text=normalized_text,
                reference_audio=ref_audio,
                max_new_frames=max_new_frames,
                do_sample=do_sample,
                seed=seed,
                normalizer_method=normalizer_method,
            )
    except Exception:
        return None, f"❌ Nano generation failed.\n\n{traceback.format_exc()}"


def _run_nano_onnx_fallback(
    text: str,
    reference_audio: str,
    max_new_frames: int,
    do_sample: bool,
    seed: int,
    normalizer_method: str,
) -> Tuple[Optional[str], str]:
    infer_onnx = NANO_DIR / "infer_onnx.py"
    if not infer_onnx.exists():
        return None, "❌ Nano fallback failed: missing app/MOSS-TTS-Nano/infer_onnx.py."

    with tempfile.TemporaryDirectory(prefix="moss_nano_onnx_") as tmpdir:
        out_wav = Path(tmpdir) / "nano_output.wav"
        cmd = [
            sys.executable,
            "infer_onnx.py",
            "--text",
            text.strip(),
            "--prompt-audio-path",
            reference_audio,
            "--output-audio-path",
            str(out_wav),
            "--max-new-frames",
            str(int(max_new_frames)),
            "--do-sample",
            "1" if do_sample else "0",
            "--disable-wetext-processing",
        ]
        if seed is not None and int(seed) != 0:
            cmd.extend(["--seed", str(int(seed))])

        result = subprocess.run(
            cmd,
            cwd=str(NANO_DIR),
            env={**os.environ, "PYTHONUTF8": "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            details = stderr if stderr else stdout
            return None, f"❌ Nano ONNX fallback failed (exit={result.returncode}).\n\n{details}"
        if not out_wav.exists():
            return None, "❌ Nano ONNX fallback finished but produced no output file."

        final_out = OUTPUT_DIR / f"nano_onnx_{os.getpid()}_{abs(hash(text)) % 1000000}.wav"
        out_wav.replace(final_out)
        status = "✅ Nano generation completed via ONNX fallback."
        if normalizer_method != "none":
            status += f" Text normalization: {normalizer_method}."
        return str(final_out), status


def build_nano_tab(args):
    with gr.Column():
        gr.Markdown("### 🧩 MOSS-TTS-Nano (HF Space-style Tab)")
        gr.Markdown(
            "Integrated from the Nano HF Space pattern: native model inference with reference-audio voice cloning."
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                nano_lang = gr.Dropdown(
                    choices=list(EXAMPLE_TEXTS.keys()),
                    value="English",
                    label="Example language",
                    info="Pre-fills text and sample reference audio.",
                )
                nano_text = gr.Textbox(
                    label="Text to synthesize",
                    value=EXAMPLE_TEXTS["English"],
                    lines=5,
                    placeholder="Enter text in a supported language...",
                )
                nano_reference = gr.Audio(
                    label="Reference audio (voice to clone)",
                    type="filepath",
                    sources=["upload", "microphone"],
                    value=_safe_ref_path("English", None),
                )
                with gr.Accordion("Advanced settings", open=False):
                    nano_max_frames = gr.Slider(
                        minimum=64,
                        maximum=512,
                        value=375,
                        step=16,
                        label="Max new frames",
                    )
                    nano_do_sample = gr.Checkbox(
                        value=True,
                        label="Sampling",
                    )
                    nano_seed = gr.Number(
                        value=0,
                        precision=0,
                        label="Seed (0 = random)",
                    )
                    nano_normalize = gr.Checkbox(
                        value=True,
                        label="Use wetext normalization (fallback to basic)",
                    )

                nano_generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")

            with gr.Column(scale=2):
                nano_output = gr.Audio(label="Generated speech", type="filepath")
                nano_status = gr.Textbox(label="Status", lines=14, interactive=False)

        nano_lang.change(
            fn=on_example_select,
            inputs=[nano_lang],
            outputs=[nano_text, nano_reference],
        )

        nano_generate_btn.click(
            fn=lambda text, ref, frames, sample, seed, norm, lang: run_nano_inference(
                text, ref, frames, sample, seed, norm, lang, args.device
            ),
            inputs=[
                nano_text,
                nano_reference,
                nano_max_frames,
                nano_do_sample,
                nano_seed,
                nano_normalize,
                nano_lang,
            ],
            outputs=[nano_output, nano_status],
        )

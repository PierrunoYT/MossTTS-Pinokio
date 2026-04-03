# GGUF Integration Analysis

Analysis of adding `OpenMOSS-Team/MOSS-TTS-GGUF` support alongside the existing transformers-based models.

## Current State

The app uses `transformers` (`AutoModel`, `AutoProcessor`) for all models across 5 tabs:

| Tab | Model | Size |
|-----|-------|------|
| TTS - Voice Cloning | MOSS-TTS (8B) / MOSS-TTS-Local (1.7B) | 8B / 1.7B |
| TTSD - Dialogue | MOSS-TTSD-v1.0 | 8B |
| Voice Generator | MOSS-VoiceGenerator | 1.7B |
| Sound Effects | MOSS-SoundEffect | 8B |
| Realtime TTS | MOSS-TTS-Realtime | 1.7B |

Models are loaded on-demand from HuggingFace using `AutoModel.from_pretrained()`.

## GGUF Model Requirements

From [OpenMOSS-Team/MOSS-TTS-GGUF](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-GGUF):

- Needs `llama-cpp-python` bindings
- Needs a companion **ONNX audio tokenizer** (`OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX`)
- Requires a **C bridge** to be built from the MOSS-TTS source repo
- Installed via `pip install -e ".[llama-cpp-onnx]"` from the MOSS-TTS GitHub repo
- Invoked via `python -m moss_tts_delay.llama_cpp --config ... --text ... --output ...`

### Available Quantizations

| Quantization | Size | EN WER ↓ | EN SIM ↑ | ZH CER ↓ | ZH SIM ↑ |
|---|---|---|---|---|---|
| Baseline (HuggingFace) | ~16 GB | 1.79% | 71.46% | 1.32% | 77.05% |
| Q8_0 | 8.74 GB | 3.21% | 68.61% | 1.56% | 76.03% |
| Q6_K | 6.75 GB | 3.11% | 68.77% | 1.44% | 76.06% |
| Q5_K_M | 5.87 GB | 2.95% | 68.55% | 1.50% | 75.96% |
| Q4_K_M | 5.05 GB | 2.83% | 68.15% | 1.58% | 75.71% |

## Feasibility

**Yes, it can be added**, but with the following constraints:

- Only the **TTS tab** can use GGUF — it is the only model in the family with GGUF weights (`MossTTSDelay-8B`)
- TTSD, VoiceGenerator, SoundEffect, and Realtime have no GGUF variants and remain transformers-only
- It would appear as a **third model variant** in the TTS tab, alongside the existing 8B and 1.7B options

## What Would Change

### `app.py`

- Add `MOSS-TTS GGUF (Q4_K_M)` (and optionally other quant levels) to the model variant radio in the TTS tab
- Add a separate inference function that calls the llama.cpp pipeline instead of `AutoModel.generate()`
- The llama.cpp backend uses a different API — either subprocess calls to `python -m moss_tts_delay.llama_cpp` or direct import of the pipeline class from the installed package

### `install.js`

Two additional download steps:

```bash
huggingface-cli download OpenMOSS-Team/MOSS-TTS-GGUF --local-dir weights/MOSS-TTS-GGUF
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX --local-dir weights/MOSS-Audio-Tokenizer-ONNX
```

And installation of the llama.cpp extras:

```bash
pip install -e ".[llama-cpp-onnx]"
```

Note: this requires the MOSS-TTS package to be installed in editable mode from the cloned source repo, not from PyPI.

### `requirements.txt`

Additional dependencies:

```
llama-cpp-python
onnxruntime  # or onnxruntime-gpu for NVIDIA
```

## Key Concerns

### 1. llama.cpp Python API vs CLI

The GGUF backend is documented as a CLI tool (`python -m moss_tts_delay.llama_cpp`). Wrapping it in a Gradio UI requires either:
- **Subprocess calls** — straightforward but adds latency and complicates audio return
- **Direct import** — cleaner, but requires studying the internal pipeline class API from the source repo

Resolution: fetch the [llama.cpp backend README](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_delay/llama_cpp/README.md) and the source to determine the importable Python API before writing any code.

### 2. C Bridge Build Step

The llama.cpp backend requires building a C bridge. This step:
- May require a C compiler (`gcc` / `cl.exe` on Windows)
- Could fail silently on machines without build tools
- Needs to be included in `install.js` and verified cross-platform

### 3. ONNX Runtime vs TensorRT

The quick-start uses ONNX Runtime (cross-platform). TensorRT is NVIDIA-only and significantly faster but more complex to install. Recommended approach: use ONNX Runtime by default, with TensorRT as an optional advanced path.

### 4. Model Download Size

The GGUF weights (`Q4_K_M`) are ~5 GB. The ONNX tokenizer adds additional size. This must be clearly communicated to users before download.

### 5. HuggingFace Gated Access

The MOSS-TTS-GGUF repo requires agreeing to share contact information. Users must be logged in to HuggingFace CLI (`huggingface-cli login`) before the download step will work. The install script needs to handle or surface this error gracefully.

## Recommended Implementation Plan

1. Fetch and read the [llama.cpp backend source/README](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_delay/llama_cpp/README.md) to identify the Python-importable pipeline API
2. Add GGUF model variant option to the TTS tab radio in `app.py`
3. Implement `run_tts_gguf_inference()` using the pipeline API (or subprocess fallback)
4. Add GGUF weight download steps to `install.js` (optional/separate install step to avoid forcing all users to download 5+ GB)
5. Add `onnxruntime` to `requirements.txt`
6. Update `README.md` with GGUF section and HuggingFace login requirement

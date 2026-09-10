# MOSS-TTS Unified Interface

A unified web interface combining all MOSS-TTS models into a single application with an intuitive tabbed interface.

## Features

**Voice Synthesis Models**
- **MOSS-TTS-v1.5** - High-fidelity voice cloning (31 languages, pause markers, improved cloning)
- **MOSS-TTSD** - Multi-speaker dialogue generation
- **MOSS-VoiceGenerator** - Design voices from text descriptions
- **MOSS-SoundEffect** - Generate environmental sounds and effects
- **MOSS-TTS-Realtime** - Low-latency streaming TTS for voice agents

**Interface**
- Single unified interface with tab-based navigation
- Smart on-demand model loading to optimize memory usage
- Modern, responsive UI built with Gradio

## Quick Start with Pinokio

This application is packaged for [Pinokio](https://pinokio.com/) for one-click installation and management.

**Available Commands:**
- **Install** - Sets up Python environment, installs dependencies, and configures PyTorch for your GPU
- **Start** - Launches the Gradio UI on `127.0.0.1` using Pinokio’s next available port (avoiding conflicts). After startup, use **Open Web UI** in Pinokio or the URL printed in the terminal.
- **Update** - Pulls launcher and model-source changes, restores missing source repositories, and refreshes dependencies without starting the server. Local Git conflicts stop the update and are shown in the terminal.
- **Reset** - Removes the `env` virtual environment for a clean reinstall

## Programmatic access

- **Pinokio (launcher scripts)** — Run actions from the Pinokio UI, or invoke scripts programmatically with Pinokio’s `script.start` API (for example from another launcher script) using the target script’s file name as `uri` and any `params` your flow needs. Launcher scripts live in the project root (`install.js`, `start.js`, `update.js`, `reset.js`, `link.js`).
- **Python** — Application code lives under `app/`. After dependencies are installed (virtualenv at project root), from the repo root:  
  `python app/app.py --host 127.0.0.1 --port <port>`  
  Or: `cd app` then `python app.py ...`. Defaults are `--host 127.0.0.1` and `--port 7860`. `GRADIO_SERVER_NAME` and `GRADIO_SERVER_PORT` (or `PORT`) supply defaults; explicit CLI flags take precedence. `--model_path` overrides the main TTS checkpoint.
- **HTTP (curl)** — The Gradio server is a normal HTTP app. Once it is listening, you can probe it with curl, for example:  
  `curl -sS -I http://127.0.0.1:<port>/`  
  Replace `<port>` with the port shown at startup (Pinokio assigns a free port when you start from the launcher).

## System Requirements

**Minimum:**
- Python 3.10 (Pinokio's bundled Flash Attention wheels target Python 3.10)
- 16GB RAM
- 50GB free disk space
- Internet connection for model downloads

**Recommended:**
- 32GB RAM
- NVIDIA GPU with 10GB+ VRAM (24GB+ for optimal performance)
- CUDA 12.8 compatible drivers

**Note:** CPU-only mode is supported but significantly slower.

Windows and Linux are supported, along with Apple Silicon macOS in CPU mode.
AMD GPUs on Windows use CPU mode because this app does not implement DirectML.
Intel macOS is unsupported: its available PyTorch 2.2 wheels do not meet the
Transformers 5 requirement of PyTorch 2.4 or newer.

## Usage Guide

### Voice Cloning (TTS)

Generate speech with optional voice cloning from reference audio (default: **MOSS-TTS-v1.5**).

1. Enter your text (use `[pause 3.2s]` in text for explicit pauses on v1.5)
2. For non Chinese/English, pick a **Language tag** when using the 8B model
3. Optionally upload reference audio (3-30 seconds recommended)
4. Adjust generation settings if needed
5. Click "Generate Speech"

**Without reference audio:** Uses default voice  
**With reference audio:** Clones the voice characteristics

### Dialogue Generation (TTSD)

Create multi-speaker conversations with distinct voices.

1. Write your dialogue with speaker tags:
```
[S1] Hello there!
[S2] Hi! How are you?
[S1] Great weather today.
```
2. Set the number of speakers (1-5)
3. Click "Generate Dialogue"

### Voice Design (VoiceGenerator)

Create custom voices from text descriptions without reference audio.

1. Describe the desired voice characteristics:
   - Age and gender
   - Tone and emotion
   - Accent or style
2. Enter text to synthesize
3. Click "Generate Voice"

**Example descriptions:**
- "A young female with a cheerful, energetic tone"
- "An elderly male with a calm, wise voice"
- "A middle-aged professional with a confident tone"

### Sound Effects

Generate environmental sounds and audio effects from descriptions.

1. Describe the sound you want:
   - "Thunder and rain in a storm"
   - "Busy city street with traffic"
   - "Crackling fireplace"
2. Click "Generate Sound"

## Generation Settings

- **Temperature** (0.1-3.0): Controls randomness. Lower = more stable, higher = more creative
- **Top P** (0.1-1.0): Nucleus sampling threshold for token selection
- **Top K** (1-200): Limits vocabulary selection to top K tokens
- **Max New Tokens**: Controls maximum output length

## Memory Usage

| Model | VRAM (bf16) | VRAM (4-bit) |
|-------|-------------|--------------|
| MOSS-TTS-v1.5 (8B) | ~16GB | ~6GB |
| MOSS-TTSD (8B) | ~16GB | ~6GB |
| MOSS-VoiceGenerator | ~8GB | ~3GB |
| MOSS-SoundEffect (8B) | ~16GB | ~6GB |
| MOSS-TTS-Realtime (1.7B) | ~4GB | Not implemented |

Figures are weights only; generation adds a KV cache that grows with
`max_new_tokens`.

**Only one model stays resident in GPU memory at a time.** When you switch tabs
(or generate with a different model) the previous model's VRAM is freed before
the new one loads, so a single 24GB card is plenty for any individual model.
If you exceed VRAM, the NVIDIA driver silently spills to system RAM and
generation becomes extremely slow — keeping one model resident avoids that.
Nano's native model also participates in this cache. All six generation buttons
share one queue so requests from different tabs cannot load models concurrently.

### Tuning memory & speed

| Setting | CLI flag | Env var | Default |
|---------|----------|---------|---------|
| Weight quantization | `--quantization {auto,none,8bit,4bit}` | `MOSS_TTS_QUANTIZATION` | `auto` (Pinokio) |
| Models kept in VRAM | `--model_cache_size N` | `MOSS_TTS_MODEL_CACHE_SIZE` | `1` |

- **`--quantization auto`** (the Pinokio default) loads 4-bit weights on CUDA
  cards with ≤32GB VRAM when `bitsandbytes` is installed, and falls back to bf16
  otherwise. This keeps the 8B SFX/Dialogue models around ~6GB so they never
  overflow a 24GB card.
- **`--quantization 4bit`** forces the same 4-bit path (needs
  `pip install bitsandbytes`, CUDA only). Use `none` to force full bf16.
- **`--model_cache_size`** trades VRAM for speed: raise it on a large-VRAM card
  to avoid reloading a model each time you revisit its tab; keep it at `1` on a
  24GB card.

## Troubleshooting

**Out of Memory / very slow generation** (especially SFX & Dialogue)
- 4-bit quantization is on by default (`auto`) so the 8B models fit in ~6GB.
  If generation is still slow, the most common cause on Windows is below.
- **On Windows, disable the NVIDIA driver's *CUDA - System Memory Fallback***:
  NVIDIA Control Panel → Manage 3D Settings → *CUDA - Sysmem Fallback Policy* →
  **Prefer No Sysmem Fallback**. When VRAM is exceeded this driver feature
  silently spills to system RAM over PCIe, making generation 10–100× slower
  (the classic "fills VRAM and becomes glacial" symptom) instead of erroring.
- Confirm `bitsandbytes` installed (`auto` falls back to bf16 without it — check
  the startup log for the `[quantization] auto:` line)
- Close other GPU applications
- Reduce `max_new_tokens` / target duration
- Use CPU mode if GPU memory is insufficient

**Installation Issues**
- Run **Reset** in Pinokio
- Run **Install** again
- Check Python version compatibility

**Model Download Failures**
- Verify internet connection
- Ensure sufficient disk space (~50GB)
- Check firewall settings

**Poor Audio Quality**
- Use high-quality reference audio (clear, minimal background noise)
- Adjust temperature setting (try 0.7-1.0 range)
- Ensure reference audio is 3-30 seconds long

## Best Practices

1. **Test incrementally** - Start with short text to verify settings
2. **Quality reference audio** - Use clear recordings with minimal background noise
3. **Descriptive prompts** - Provide detailed descriptions for voice/sound generation
4. **Adjust settings** - Experiment with temperature and sampling parameters
5. **Monitor memory** - Close unused applications when running large models

## Development checks

Install the application requirements plus `pytest` in a Python environment with
PyTorch, then run from the repository root:

```bash
python -m pytest tests -q
node --test tests/launchers.test.js
```

The tests cover UI construction, generation queue sharing, device and quantization
selection, input validation, audio file handling, and launcher control flow. Model
loaders are mocked where weights would otherwise be downloaded; these checks do
not measure speech quality or validate CUDA/ROCm inference.

## Resources

- **Source Code**: [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)
- **Issues**: [GitHub Issues](https://github.com/OpenMOSS/MOSS-TTS/issues)
- **Community**: [OpenMOSS Discord](https://discord.gg/fvm5TaWjU3)

## License

Apache 2.0 License (same as MOSS-TTS)

## Acknowledgments

- **OpenMOSS Team** - MOSS-TTS model development
- **MOSI.AI** - Research and model training
- **Gradio** - Web interface framework

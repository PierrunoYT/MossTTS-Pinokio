# MOSS-TTS Unified Interface

🎵 **All-in-One Gradio UI for the MOSS-TTS Family**

A unified web interface that combines all MOSS-TTS models into a single application with tabs.

## 🌟 Features

- **🎙️ MOSS-TTS** - High-fidelity voice cloning with zero-shot capability
- **💬 MOSS-TTSD** - Multi-speaker dialogue generation
- **🎨 MOSS-VoiceGenerator** - Design voices from text descriptions
- **🔊 MOSS-SoundEffect** - Generate environmental sounds and effects
- **📱 Single Interface** - All models accessible through tabs
- **⚡ Smart Model Loading** - Models load on-demand to save memory
- **🎨 Modern UI** - Clean, intuitive interface built with Gradio

---

## 🚀 Pinokio (One-Click Install)

This app is packaged for [Pinokio](https://pinokio.com/) for one-click installation:

1. **Install** - Creates Python venv, installs dependencies, configures PyTorch for your GPU (NVIDIA/AMD/Apple Silicon)
2. **Start** - Launches the Gradio UI at http://localhost:7860
3. **Update** - Pulls latest changes via `git pull`
4. **Reset** - Removes the environment (re-run Install to start fresh)
5. **Save Disk Space** - Deduplicates shared library files across Pinokio apps

---

## 📋 Prerequisites

### System Requirements

- **Python**: 3.10 or higher (3.12 recommended)
- **RAM**: Minimum 16GB (32GB recommended)
- **Disk Space**: ~50GB free space (for all models and dependencies)
- **GPU** (optional but recommended):
  - NVIDIA GPU with CUDA support
  - CUDA 12.8 compatible drivers
  - Minimum 10GB VRAM (24GB+ recommended)

### Software Requirements

- **[Pinokio](https://pinokio.com/)** - One-click app manager
- **CUDA drivers** (optional, for NVIDIA GPU acceleration)

---

## 🎮 Usage

Click **Start** in Pinokio, then open your browser to **http://localhost:7860**

---

## 📖 How to Use Each Tab

### 🎙️ TTS - Voice Cloning

1. Enter text to synthesize
2. (Optional) Upload reference audio to clone a voice
3. Click "Generate Speech"

**Tips:** Without reference = default voice. With reference = voice cloning. Works with any language.

### 💬 TTSD - Dialogue Generation

1. Enter dialogue script with speaker labels:
   ```
   Speaker1: Hello there!
   Speaker2: Hi! How are you?
   ```
2. Set number of speakers (1-5)
3. Click "Generate Dialogue"

### 🎨 Voice Generator

1. Describe the voice (e.g., "A young female with a cheerful tone")
2. Enter text to synthesize
3. Click "Generate Voice"

**Tips:** Be descriptive about age, gender, tone. No reference audio needed.

### 🔊 Sound Effects

1. Describe the sound (e.g., "Thunder and rain in a storm")
2. Click "Generate Sound"

---

## ⚙️ Settings

- **Temperature** (0.1-3.0): Higher = more creative, lower = more stable
- **Top P** (0.1-1.0): Nucleus sampling threshold
- **Top K** (1-200): Limits token selection
- **Max New Tokens**: Maximum audio length

---

## 🔧 Troubleshooting

### CUDA Out of Memory

- Close other GPU apps
- Reduce `max_new_tokens` in the UI

### Install / Import Errors

Run **Reset** in Pinokio, then **Install** again.

### Models Not Loading

- Check internet connection
- Ensure ~50GB disk space

---

## 📊 Memory Requirements

| Model | GPU VRAM |
|-------|----------|
| MOSS-TTS | ~10GB |
| MOSS-TTSD | ~10GB |
| MOSS-VoiceGenerator | ~8GB |
| MOSS-SoundEffect | ~10GB |

Models load on-demand; only the active model uses memory.

---

## 🎯 Best Practices

1. **Start small** - Test with short text first
2. **Good reference audio** - Clear, high-quality for voice cloning
3. **Be descriptive** - Detailed descriptions for voice/sound generation
4. **Experiment** - Try different settings

---

## 🆘 Getting Help

- **GitHub Issues**: [MOSS-TTS Issues](https://github.com/OpenMOSS/MOSS-TTS/issues)
- **Documentation**: [MOSS-TTS Docs](https://github.com/OpenMOSS/MOSS-TTS)
- **Discord**: [OpenMOSS Discord](https://discord.gg/fvm5TaWjU3)

---

## 📄 License

Apache 2.0 (same as MOSS-TTS)

## 🙏 Credits

- **OpenMOSS Team** - MOSS-TTS models
- **MOSI.AI** - Model development
- **Gradio** - Web interface framework

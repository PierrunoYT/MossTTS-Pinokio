"""Info / About tab."""

import gradio as gr


def build_info_tab():
    with gr.Column():
        gr.Markdown("""
# 🎵 MOSS-TTS Unified Interface

Welcome to the all-in-one interface for the MOSS-TTS Family of models!

## 📚 Available Models

### 🎙️ MOSS-TTS
High-fidelity text-to-speech with zero-shot voice cloning. Upload a reference audio to clone any
voice! Choose between **MOSS-TTS (8B)** for best long-form stability and
**MOSS-TTS-Local (1.7B)** for the highest speaker similarity score with lower VRAM usage.
Supports **Clone**, **Continuation**, and **Continuation + Clone** modes.

### 💬 MOSS-TTSD
Multi-speaker dialogue generation for creating realistic conversations with different voices.

### 🎨 MOSS-VoiceGenerator
Design custom voices from text descriptions without needing reference audio.

### 🔊 MOSS-SoundEffect
Generate environmental sounds and effects from text descriptions.

### ⚡ MOSS-TTS-Realtime
Low-latency streaming TTS optimised for real-time voice agents.
Achieves ~180 ms TTFB after warm-up (1.7B model).

### 🗣️ MOSS-Speech
True speech-to-speech language model (9B) that works without text guidance.
Supports all four interaction modes: Speech→Speech, Speech→Text, Text→Speech, Text→Text.
Preserves paralinguistic cues like tone, emotion, and prosody. Bilingual (Chinese/English).

## 🚀 Quick Start

1. **Choose a tab** for the model you want to use
2. **Enter your text** or description
3. **Adjust settings** if needed (optional)
4. **Click Generate** and wait for the result

## ⚙️ Tips

- Adjust **Temperature** for creativity vs. stability (higher = more expressive)
- Use **reference audio** in MOSS-TTS for voice cloning
- In **Continuation** modes, prepend the reference audio transcript to your input text
- Be descriptive in voice/sound descriptions for better results
- Generation time depends on text length and your hardware

## 📖 Learn More

- [GitHub Repository](https://github.com/OpenMOSS/MOSS-TTS)
- [MOSS-TTS Collection](https://huggingface.co/collections/OpenMOSS-Team/moss-tts)
- [MOSS-TTSD Collection](https://huggingface.co/collections/OpenMOSS-Team/moss-ttsd)
- [MOSS-Speech Collection](https://huggingface.co/collections/OpenMOSS-Team/moss-speech)
- [MOSI.AI Website](https://mosi.cn/models/moss-tts)

## 📝 License

All models are released under Apache 2.0 License.

---

**Note:** First-time generation may take longer as models are downloaded and loaded.
        """)

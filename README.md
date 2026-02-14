# MOSS-TTS Unified Interface

A unified web interface combining all MOSS-TTS models into a single application with an intuitive tabbed interface.

## Features

**Voice Synthesis Models**
- **MOSS-TTS** - High-fidelity voice cloning with zero-shot capability
- **MOSS-TTSD** - Multi-speaker dialogue generation
- **MOSS-VoiceGenerator** - Design voices from text descriptions
- **MOSS-SoundEffect** - Generate environmental sounds and effects

**Interface**
- Single unified interface with tab-based navigation
- Smart on-demand model loading to optimize memory usage
- Modern, responsive UI built with Gradio

## Quick Start with Pinokio

This application is packaged for [Pinokio](https://pinokio.com/) for one-click installation and management.

**Available Commands:**
- **Install** - Sets up Python environment, installs dependencies, and configures PyTorch for your GPU
- **Start** - Launches the Gradio UI at `http://localhost:7860`
- **Update** - Pulls latest changes from repository
- **Reset** - Removes environment for clean reinstall

## System Requirements

**Minimum:**
- Python 3.10 or higher (3.12 recommended)
- 16GB RAM
- 50GB free disk space
- Internet connection for model downloads

**Recommended:**
- 32GB RAM
- NVIDIA GPU with 10GB+ VRAM (24GB+ for optimal performance)
- CUDA 12.8 compatible drivers

**Note:** CPU-only mode is supported but significantly slower.

## Usage Guide

### Voice Cloning (TTS)

Generate speech with optional voice cloning from reference audio.

1. Enter your text
2. Optionally upload reference audio (3-30 seconds recommended)
3. Adjust generation settings if needed
4. Click "Generate Speech"

**Without reference audio:** Uses default voice  
**With reference audio:** Clones the voice characteristics

### Dialogue Generation (TTSD)

Create multi-speaker conversations with distinct voices.

1. Write your dialogue with speaker labels:
```
Speaker1: Hello there!
Speaker2: Hi! How are you?
Speaker1: Great weather today.
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

| Model | VRAM Required |
|-------|---------------|
| MOSS-TTS | ~10GB |
| MOSS-TTSD | ~10GB |
| MOSS-VoiceGenerator | ~8GB |
| MOSS-SoundEffect | ~10GB |

Models load on-demand. Only the active tab's model occupies memory.

## Troubleshooting

**Out of Memory Errors**
- Close other GPU applications
- Reduce `max_new_tokens` setting
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

module.exports = {
  run: [
    {
      when: "{{platform === 'darwin' && arch !== 'arm64'}}",
      method: "notify",
      params: {
        html: "Intel macOS is unsupported: MOSS-TTS requires PyTorch 2.4 or newer. Use Windows, Linux, or an Apple Silicon Mac."
      },
      next: null
    },
    {
      method: "notify",
      params: {
        html: "Installing MOSS-TTS..."
      }
    },
    // Install Git LFS for large model files
    {
      method: "shell.run",
      params: {
        message: "git lfs install"
      }
    },
    // Install MOSS-TTS and dependencies
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install -r app/requirements.txt"
        ],
      }
    },
    // Install PyTorch with GPU support + Flash Attention for NVIDIA
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: ".",
          flashattention: true,
          triton: true
        }
      }
    },
    // Clone MOSS-TTS repo (needed for mossttsrealtime package)
    {
      when: "{{!exists('app/MOSS-TTS')}}",
      method: "shell.run",
      params: {
        message: "git clone https://github.com/OpenMOSS/MOSS-TTS.git app/MOSS-TTS"
      }
    },
    // Clone MOSS-TTS-Nano repo for ONNX CPU tab
    {
      when: "{{!exists('app/MOSS-TTS-Nano')}}",
      method: "shell.run",
      params: {
        message: "git clone https://github.com/OpenMOSS/MOSS-TTS-Nano.git app/MOSS-TTS-Nano"
      }
    },
    // Install mossttsrealtime package from MOSS-TTS repo
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install --no-deps -e app/MOSS-TTS"
        ],
      }
    },
    // Install Nano ONNX dependencies without overriding core stack
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install onnxruntime sentencepiece python-multipart wetext"
        ],
      }
    },
    // Install bitsandbytes for 4-bit quantization (enabled via auto mode).
    // Lets the 8B SFX/Dialogue checkpoints fit in ~5GB instead of ~16GB so
    // they don't overflow VRAM and trigger the slow NVIDIA sysmem fallback.
    {
      when: "{{gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install bitsandbytes accelerate"
        ],
      }
    },
    {
      method: "notify",
      params: {
        html: "✅ Installed! Models download on-demand (~10GB each). ~10GB VRAM or CPU."
      }
    },
    {
      when: "{{!(args && args.skip_start)}}",
      method: "script.start",
      params: {
        uri: "start.js"
      }
    }
  ]
}

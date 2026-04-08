module.exports = {
  run: [
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
      method: "shell.run",
      params: {
        message: "git clone https://github.com/OpenMOSS/MOSS-TTS.git app/MOSS-TTS || true"
      }
    },
    // Install mossttsrealtime package from MOSS-TTS repo
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "pip install --no-deps -e app/MOSS-TTS"
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
      method: "script.start",
      params: {
        uri: "start.js"
      }
    }
  ]
}

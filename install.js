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
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
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
          path: "app",
          flashattention: true,
          triton: true
        }
      }
    },
    // Create link for UI
    {
      method: "script.start",
      params: {
        uri: "link.js"
      }
    },
    {
      method: "notify",
      params: {
        html: "✅ Installed! Models download on-demand (~10GB each). ~10GB VRAM or CPU."
      }
    }
  ]
}

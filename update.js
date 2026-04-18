module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: [
          "git pull",
          "git -C app/MOSS-TTS pull || true",
          "git -C app/MOSS-TTS-Nano pull || true"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install -r app/requirements.txt",
          "uv pip install onnxruntime sentencepiece python-multipart wetext"
        ]
      }
    },
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
    }
  ]
}

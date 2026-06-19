module.exports = {
  daemon: true,
  run: [
    {
      method: "notify",
      params: {
        html: "Starting MOSS-TTS... Models download on-demand."
      }
    },
    {
      method: "shell.run",
      params: {
        build: true,
        venv: "env",
        path: ".",
        env: {
          HF_XET_HIGH_PERFORMANCE: "1",
          HF_HUB_DOWNLOAD_TIMEOUT: "300",
          PYTHONUTF8: "1",
          TORCHDYNAMO_SUPPRESS_ERRORS: "1",
          MOSS_TTS_PRELOAD_AT_STARTUP: "0",
          MOSS_TTS_QUANTIZATION: "auto"
        },
        message: [
          "python app/app.py --host 127.0.0.1 --port {{port}}"
        ],
        on: [{
          event: "/(http:\/\/\\S+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    },
    {
      method: "notify",
      params: {
        html: "✅ MOSS-TTS running! Voice cloning • Dialogue • Sound effects"
      }
    }
  ]
}

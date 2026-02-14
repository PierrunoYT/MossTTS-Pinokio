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
        env: {
          GRADIO_SERVER_NAME: "127.0.0.1",
          GRADIO_SERVER_PORT: "7860",
          HF_HUB_ENABLE_HF_TRANSFER: "1",
          PYTHONUTF8: "1",
          TORCHDYNAMO_SUPPRESS_ERRORS: "1"
        },
        message: [
          "python app.py"
        ],
        on: [{
          event: "/http:\/\/\\S+/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[0]}}"
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


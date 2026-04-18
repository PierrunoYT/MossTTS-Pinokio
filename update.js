module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: [
        "git pull",
        "git -C app/MOSS-TTS pull || true",
        "git -C app/MOSS-TTS-Nano pull || true"
      ]
    }
  }]
}

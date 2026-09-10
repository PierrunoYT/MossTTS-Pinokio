module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull --ff-only"
      }
    },
    ...["app/MOSS-TTS", "app/MOSS-TTS-Nano"].map(path => ({
      when: `{{exists('${path}')}}`,
      method: "shell.run",
      params: {
        path,
        message: "git pull --ff-only"
      }
    })),
    // Reuse installation so missing repos and all dependencies are restored.
    {
      method: "script.start",
      params: {
        uri: "install.js",
        params: { skip_start: true }
      }
    }
  ]
}

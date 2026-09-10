from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from core import audio, download, memory
from tabs import nano


def test_reference_truncation_writes_closed_temp_file(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros(2000), 1000)
    result = Path(audio.truncate_reference_audio(str(source), max_duration=1))
    try:
        assert result != source
        assert sf.info(result).frames == 1000
        assert sf.info(source).frames == 2000
    finally:
        result.unlink()


def test_reference_write_failure_removes_temporary_file(monkeypatch, tmp_path):
    import librosa

    monkeypatch.setattr(librosa, "load", lambda *a, **k: (np.zeros(2000), 1000))
    monkeypatch.setattr(audio.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(sf, "write", MagicMock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        audio.truncate_reference_audio("reference.wav", max_duration=1)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_downloads_and_loading_share_local_cache(monkeypatch, tmp_path, platform):
    monkeypatch.setattr(download.sys, "platform", platform)
    monkeypatch.setattr(download, "_download_with_retries", lambda repo: str(tmp_path / repo.replace("/", "__")))
    assert download.resolve_hf_path("Org/Model") == str(tmp_path / "Org__Model")
    assert download.resolve_hf_path(str(tmp_path)) == str(tmp_path)


def test_nano_fallback_outputs_are_unique_and_cross_drive_safe(monkeypatch, tmp_path):
    (tmp_path / "infer_onnx.py").touch()
    monkeypatch.setattr(nano, "NANO_DIR", tmp_path)
    monkeypatch.setattr(nano, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(Path, "replace", MagicMock(side_effect=OSError("cross-device link")))

    def run(cmd, **kwargs):
        assert kwargs["encoding"] == "utf-8"
        Path(cmd[cmd.index("--output-audio-path") + 1]).write_bytes(b"audio")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(nano.subprocess, "run", run)
    args = ("same text", "ref.wav", 100, True, 1, "none")
    first, _ = nano._run_nano_onnx_fallback(*args)
    second, _ = nano._run_nano_onnx_fallback(*args)
    assert first != second
    assert Path(first).read_bytes() == Path(second).read_bytes() == b"audio"


def test_nano_shares_eviction_budget(monkeypatch):
    monkeypatch.setattr(memory, "_RESIDENTS", memory.OrderedDict())
    monkeypatch.setenv("MOSS_TTS_MODEL_CACHE_SIZE", "1")
    monkeypatch.setattr(nano, "_ensure_nano_repo", lambda: None)
    monkeypatch.setattr(nano, "resolve_hf_path", lambda repo: repo)
    for loader in (nano.AutoModelForCausalLM, nano.AutoModel, nano.AutoTokenizer):
        monkeypatch.setattr(loader, "from_pretrained", MagicMock())
    memory.cache_put(("old",), (object(),))
    loaded = nano._load_nano_runtime("cpu")
    assert memory.cache_get(("old",)) is None
    assert nano._load_nano_runtime("cpu") is loaded
    memory.unload_all_models()
    assert nano._load_nano_runtime("cpu") is not loaded
    memory.unload_all_models()

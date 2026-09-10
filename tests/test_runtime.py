from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from core import engine, memory, quant, runtime


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.setattr(quant, "_OVERRIDE", None)
    monkeypatch.setattr(memory, "_RESIDENTS", memory.OrderedDict())
    monkeypatch.setenv("MOSS_TTS_MODEL_CACHE_SIZE", "1")


def test_auto_quantization_honors_selected_device(monkeypatch):
    monkeypatch.setenv("MOSS_TTS_QUANTIZATION", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(quant, "bitsandbytes_available", lambda: True)
    seen = []

    def properties(device):
        seen.append(device)
        return SimpleNamespace(total_memory=8 * 1024**3)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    assert quant.resolve_quantization(torch.device("cpu")) == "none"
    assert not seen
    assert quant.resolve_quantization(torch.device("cuda:1")) == "4bit"
    assert seen == [torch.device("cuda:1")]


def test_auto_override_does_not_restore_explicit_env(monkeypatch):
    monkeypatch.setenv("MOSS_TTS_QUANTIZATION", "8bit")
    monkeypatch.setattr(quant, "_auto_quantization", lambda device=None: "4bit")
    assert quant.set_quantization_override("auto") == "4bit"
    assert quant.set_quantization_override(None) == "8bit"


def test_dtype_falls_back_on_older_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert runtime.pick_dtype(torch.device("cuda:1")) == torch.float16
    assert runtime.pick_dtype(torch.device("cpu")) == torch.float32
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert runtime.pick_dtype(torch.device("cuda:1")) == torch.bfloat16


@pytest.mark.parametrize("empty", [False, True])
def test_generation_disables_grad_for_processing_and_decode(monkeypatch, empty):
    cleaned = []
    monkeypatch.setattr(engine, "free_after_generation", cleaned.append)

    class Processor:
        def __call__(self, conversations, mode):
            assert not torch.is_grad_enabled()
            return {"input_ids": torch.ones(1), "attention_mask": torch.ones(1)}

        def decode(self, outputs):
            assert not torch.is_grad_enabled()
            return [SimpleNamespace(audio_codes_list=[] if empty else [torch.zeros(10)])]

    class Model:
        def generate(self, **kwargs):
            assert not torch.is_grad_enabled()
            return torch.ones(1)

    call = lambda: engine.generate_and_decode(
        Model(), Processor(), torch.device("cpu"), [], "generation", engine.Sampling(), 10
    )
    if empty:
        with pytest.raises(RuntimeError, match="decodable"):
            call()
    else:
        assert call().shape == (10,)
    assert cleaned == [torch.device("cpu")]


def test_ui_constructs_and_serializes_all_generators():
    import app

    ui = app.build_unified_interface(SimpleNamespace(device="cpu", attn_implementation="eager"))
    try:
        generators = [f for f in ui.fns.values() if f.concurrency_id == "model_inference"]
        assert len(generators) == 6
        assert all(f.concurrency_limit == 1 for f in generators)
    finally:
        ui.close()


def test_quantization_dropdown_selects_auto(monkeypatch):
    import app

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(app, "bitsandbytes_available", lambda: True)
    monkeypatch.setattr(app, "resolve_quantization", lambda: "8bit")
    selected = []
    monkeypatch.setattr(app, "set_quantization_override", lambda mode: selected.append(mode) or "4bit")
    with app.gr.Blocks() as ui:
        app._build_quantization_control()
    try:
        callback = next(f.fn for f in ui.fns.values() if f.fn.__name__ == "_on_change")
        callback(app._QUANT_LABELS["auto"])
        assert selected == ["auto"]
    finally:
        ui.close()


def test_cli_overrides_environment_and_applies_model_path(monkeypatch):
    import app

    monkeypatch.setattr(app.sys, "argv", ["app.py", "--host", "127.0.0.1", "--port", "8123", "--model_path", "custom/model"])
    monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")
    monkeypatch.setenv("GRADIO_SERVER_PORT", "9000")
    monkeypatch.setenv("MOSS_TTS_PRELOAD_AT_STARTUP", "0")
    monkeypatch.setattr(app, "MODELS", app.MODELS.copy())
    captured = {}

    class UI:
        def queue(self, **kwargs):
            return self

        def launch(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(app, "build_unified_interface", lambda args: UI())
    app.main()
    assert captured["server_name"] == "127.0.0.1"
    assert captured["server_port"] == 8123
    assert "css" in captured and "theme" in captured
    assert app.MODELS["tts"] == "custom/model"

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tabs import sound_effect, tts, ttsd, voice_gen
from utils import parse_port


@pytest.mark.parametrize("tab,call_args", [
    (sound_effect, ("Rain", 2, 1, 0.9, 50, 1, 100, "cpu", "eager")),
    (voice_gen, ("Calm voice", "Hello", 1, 0.9, 50, 1, 100, "cpu", "eager")),
])
def test_non_tts_generators_send_batched_conversations(monkeypatch, tab, call_args):
    message = object()
    processor = SimpleNamespace(build_user_message=lambda **kwargs: message)
    monkeypatch.setattr(tab, "load_model", lambda *args: (None, processor, torch.device("cpu"), 24000))

    def generate(model, processor, device, conversations, *args):
        assert conversations == [[message]]
        return np.zeros(10, dtype=np.int16)

    monkeypatch.setattr(tab, "generate_and_decode", generate)
    function = tab.run_sound_effect_inference if tab is sound_effect else tab.run_voice_gen_inference
    audio, status = function(*call_args)
    assert audio is not None, status
    assert audio[0] == 24000


@pytest.mark.parametrize("dialogue", ["", "Hello", "[S0] Hello", "[S3] Hello"])
def test_invalid_dialogue_does_not_load_model(monkeypatch, dialogue):
    def forbidden(*args):
        pytest.fail("Invalid input triggered a model download/load")

    monkeypatch.setattr(ttsd, "load_model", forbidden)
    audio, status = ttsd.run_ttsd_inference(
        2, *([None] * 5), *([""] * 5), dialogue,
        True, False, 1, 0.9, 50, 1, 100, device="cpu", attn_implementation="eager",
    )
    assert audio is None
    assert "Error" in status


def test_speaker_merging_preserves_word_boundaries():
    assert ttsd.normalize_text("[S1] Hello [S1] world") == "[S1]Hello world"
    assert ttsd._merge_consecutive_speaker_tags("[S1] Hello [S1] world") == "[S1]Hello world"


def test_reference_prompt_rejects_wrong_speaker():
    with pytest.raises(ValueError, match="only contain"):
        ttsd._normalize_prompt_text("[S2] Hello", 1)
    assert ttsd._normalize_prompt_text("Hello", 1) == "[S1] Hello"


@pytest.mark.parametrize("value,expected", [("-1", 7860), ("65536", 7860), ("0", 7860), ("bad", 7860), ("8123", 8123)])
def test_port_validation(value, expected):
    assert parse_port(value, 7860) == expected


@pytest.mark.parametrize("variant,key", [(tts.TTS_VARIANT_V15, "tts"), (tts.TTS_VARIANT_LOCAL, "tts_local")])
def test_tts_download_fetches_only_selected_variant(monkeypatch, variant, key):
    requested = []
    monkeypatch.setattr(tts, "download_model_files_for_keys", lambda keys: requested.append(keys) or "ok")
    assert tts._download_tts_model(variant) == "ok"
    assert requested == [[key]]


def test_ttsd_reference_audio_is_capped(tmp_path):
    import soundfile as sf

    long_ref = tmp_path / "long.wav"
    sf.write(long_ref, np.zeros(int(ttsd.MAX_REFERENCE_DURATION_SEC * 100) + 500), 100)
    wav, sr = ttsd._load_reference_wav(str(long_ref), 1)
    assert sr == 100
    assert wav.shape == (1, int(ttsd.MAX_REFERENCE_DURATION_SEC * 100))

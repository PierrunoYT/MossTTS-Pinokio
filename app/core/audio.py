"""Audio helpers shared across tabs: reference truncation, loading, resampling,
and the float-waveform → int16 conversion that used to be copy-pasted into
every tab's inference function.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from config import MAX_REFERENCE_DURATION_SEC


def audio_to_int16(audio) -> np.ndarray:
    """Convert a model audio output (tensor or array) to a 1-D int16 waveform.

    Clipping to [-1, 1] before scaling avoids Gradio's float→int16 conversion
    warning and out-of-range wrap-around.
    """
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().float().cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)
    if audio_np.ndim > 1:
        audio_np = audio_np.reshape(-1)
    audio_np = audio_np.astype(np.float32, copy=False)
    audio_np = np.clip(audio_np, -1.0, 1.0)
    return (audio_np * 32767.0).astype(np.int16)


def truncate_reference_audio(
    audio_path: str, max_duration: float = MAX_REFERENCE_DURATION_SEC
) -> str:
    """Return ``audio_path`` unchanged if short enough, otherwise write a
    truncated copy to a temp file and return that path (prevents O(L²) OOM in
    the audio tokenizer)."""
    import librosa
    import soundfile as sf

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    max_samples = int(max_duration * sr)
    if len(y) <= max_samples:
        return audio_path

    print(
        f"⚠️  Reference audio is {len(y) / sr:.1f}s — truncating to {max_duration:.0f}s "
        f"to avoid GPU OOM in the audio tokenizer."
    )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(str(tmp_path), y[:max_samples], sr)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return str(tmp_path)


def load_audio(audio_path: str) -> Tuple[torch.Tensor, int]:
    """Load an audio file as a mono ``(1, samples)`` float32 tensor + sample rate."""
    import soundfile as sf

    path = Path(audio_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Reference audio not found: {path}")
    wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
    if wav_np.size == 0:
        raise ValueError(f"Reference audio is empty: {path}")
    if wav_np.shape[1] > 1:
        wav_np = wav_np.mean(axis=1, keepdims=True)
    return torch.from_numpy(wav_np.T), int(sr)


def resample_wav(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Linearly resample a ``(channels, samples)`` waveform."""
    if int(orig_sr) == int(target_sr):
        return wav
    new_len = int(round(wav.shape[-1] * float(target_sr) / float(orig_sr)))
    if new_len <= 0:
        raise ValueError(f"Invalid resample length from {orig_sr}Hz to {target_sr}Hz.")
    return torch.nn.functional.interpolate(
        wav.unsqueeze(0), size=new_len, mode="linear", align_corners=False
    ).squeeze(0)

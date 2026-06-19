"""Environment compatibility shims.

Keeps ``app.py`` readable by moving the transformers remote-code aliasing and
the Windows asyncio noise-suppression out of the entry point. Call
:func:`apply_compat_shims` once, early, before any model loads.
"""

from __future__ import annotations

import sys


def _alias_transformers_mappings() -> None:
    """MOSS remote code references ``MODALITY_TO_BASE_CLASS_MAPPING`` /
    ``MODALITY_TO_AE_CLASS_MAPPING``, renamed to ``AUTO_TO_BASE_CLASS_MAPPING``
    across transformers versions. Expose the mapping under every alias so the
    remote code's import doesn't crash. No-op when names already exist."""
    try:
        import transformers as _tf
        import transformers.processing_utils as _pu

        alias_names = (
            "MODALITY_TO_BASE_CLASS_MAPPING",
            "MODALITY_TO_AE_CLASS_MAPPING",
            "AUTO_TO_BASE_CLASS_MAPPING",
        )

        mapping = None
        for mod in (_pu, _tf):
            for name in alias_names:
                candidate = getattr(mod, name, None)
                if candidate is not None:
                    mapping = candidate
                    break
            if mapping is not None:
                break

        if mapping is None:
            return

        targets = [_tf, _pu]
        try:
            import transformers.models.auto as _auto
            targets.append(_auto)
        except Exception:
            pass

        for mod in targets:
            for name in alias_names:
                if not hasattr(mod, name):
                    setattr(mod, name, mapping)
    except Exception:
        pass


def _alias_pretrained_config() -> None:
    """MOSS remote code imports ``PreTrainedConfig`` from
    ``transformers.configuration_utils``; older releases only expose
    ``PretrainedConfig``. Alias both names. No-op when both exist."""
    try:
        import transformers as _tf
        import transformers.configuration_utils as _cu

        new_name, old_name = "PreTrainedConfig", "PretrainedConfig"
        cfg = getattr(_cu, new_name, None) or getattr(_cu, old_name, None)
        if cfg is None:
            return
        for mod in (_cu, _tf):
            for name in (new_name, old_name):
                if not hasattr(mod, name):
                    setattr(mod, name, cfg)
    except Exception:
        pass


def _quiet_windows_asyncio() -> None:
    """Suppress the benign WinError 10054 noise the ProactorEventLoop raises
    when a browser tab closes mid-stream."""
    if sys.platform != "win32":
        return
    try:
        from asyncio import proactor_events as _pe

        orig = _pe._ProactorBasePipeTransport._call_connection_lost  # type: ignore[attr-defined]

        def _quiet(self, exc):  # type: ignore[override]
            try:
                orig(self, exc)
            except OSError:
                pass

        _pe._ProactorBasePipeTransport._call_connection_lost = _quiet  # type: ignore[attr-defined]
    except Exception:
        pass


def apply_compat_shims() -> None:
    """Apply all compatibility shims. Safe to call multiple times."""
    _alias_transformers_mappings()
    _alias_pretrained_config()
    _quiet_windows_asyncio()

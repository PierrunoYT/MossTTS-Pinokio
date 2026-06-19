"""HuggingFace snapshot downloads and Windows-safe local path resolution."""

from __future__ import annotations

import os
import sys
import time

from config import CODEC_MODEL_PATH, MODELS

_MAX_ATTEMPTS = 5


def _local_dir_for(repo_id: str) -> str:
    """A stable, per-repo directory holding real (non-symlinked) files.

    HF's default cache stores ``snapshots/`` as symlinks into ``blobs/`` (where
    files are hash-named). transformers 5.x's ``_compute_local_source_files_hash``
    follows those symlinks and then looks for a remote-code module's relative
    imports as siblings in ``blobs/`` — which don't exist there, raising
    ``FileNotFoundError``. Downloading into a ``local_dir`` materialises real
    files side-by-side so relative imports resolve. Kept under the HF cache base
    so existing cleanup tooling sees it.
    """
    base = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    return os.path.join(base, "local", repo_id.replace("/", "__"))


def _download_with_retries(repo_id: str):
    """Download a repo into a local_dir of real files, retrying on network errors.

    Each attempt resumes thanks to the HF Hub local cache. Returns the local
    directory path.
    """
    from huggingface_hub import snapshot_download

    local_dir = _local_dir_for(repo_id)
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return snapshot_download(repo_id, local_dir=local_dir)
        except Exception as exc:
            if attempt == _MAX_ATTEMPTS:
                raise
            wait = attempt * 5
            print(
                f"⚠️  Download interrupted ({exc.__class__.__name__}: {exc}). "
                f"Retrying in {wait}s… (attempt {attempt}/{_MAX_ATTEMPTS})"
            )
            time.sleep(wait)


def resolve_hf_path(repo_id: str) -> str:
    """Resolve a HuggingFace repo ID to a local snapshot path on Windows.

    Custom processor code often calls ``Path(repo_id)`` which on Windows turns
    the ``/`` in ``Org/Model`` into backslashes, producing an invalid repo ID.
    Pre-downloading with ``snapshot_download`` gives us a real local path.
    """
    if sys.platform == "win32" and "/" in repo_id and not os.path.isdir(repo_id):
        return _download_with_retries(repo_id)
    return repo_id


def _repos_for_download(model_keys: list[str]) -> list[str]:
    """HF repo IDs needed for inference: main checkpoint(s) + shared audio tokenizer.

    The MOSS processors load the codec from ``OpenMOSS-Team/MOSS-Audio-Tokenizer``
    by default; the realtime stack loads it explicitly. Deduplicate, keep order.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for key in model_keys:
        rid = MODELS[key]
        if rid not in seen:
            seen.add(rid)
            ordered.append(rid)
    if CODEC_MODEL_PATH not in seen:
        ordered.append(CODEC_MODEL_PATH)
    return ordered


def download_model_files_for_keys(model_keys: list[str]) -> str:
    """Download all repos used when loading these model keys (main + codec)."""
    repos = _repos_for_download(model_keys)
    for repo_id in repos:
        print(f"Downloading {repo_id}…")
        _download_with_retries(repo_id)
        print(f"✓ {repo_id} downloaded")
    return f"✅ Downloaded successfully: {', '.join(repos)}"


def download_model_files(model_key: str) -> str:
    """Download a single tab's checkpoint + the shared codec to the HF cache."""
    return download_model_files_for_keys([model_key])

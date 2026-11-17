"""Explicit, integrity-checked model installation; no implicit inference downloads."""
from __future__ import annotations

import hashlib
import json
import os
from importlib.resources import files
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen


class ModelStoreError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def load_manifest() -> dict:
    return json.loads(files("biometry_parser").joinpath("model_manifest.json").read_text(encoding="utf-8"))


def default_cache_dir() -> Path:
    return Path(os.environ.get("BIOMETRY_PARSER_MODEL_CACHE", str(Path.home() / ".cache" / "biometry_parser" / "models")))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model(name: str, manifest: dict | None = None) -> dict:
    try:
        return (manifest or load_manifest())["models"][name.lower()]
    except KeyError:
        raise ModelStoreError("model_unavailable", "Model is not in the pinned manifest; provide explicit local model paths.") from None


def resolve_model(name: str, cache_dir: str | Path | None = None, *, manifest: dict | None = None) -> dict[str, str]:
    name = name.lower()
    spec = _model(name, manifest)
    base = Path(cache_dir) if cache_dir else default_cache_dir()
    result = {}
    hashes = []
    for role, artifact in spec["artifacts"].items():
        path = base / name / artifact["filename"]
        if not path.is_file():
            raise ModelStoreError("model_unavailable", "Pinned model files are missing; run models download first.")
        actual = file_sha256(path)
        if actual != artifact["sha256"]:
            raise ModelStoreError("model_hash_mismatch", "A model artifact failed its SHA-256 integrity check.")
        result[role] = str(path.resolve())
        hashes.append(actual)
    result["hash"] = hashlib.sha256("".join(hashes).encode()).hexdigest()
    result["version"] = spec.get("version", "")
    result["size"] = spec.get("size", "")
    result["recognition_language"] = spec.get("recognition_language", "ch")
    return result


def download_model(name: str, cache_dir: str | Path | None = None, *, manifest: dict | None = None, timeout: float = 60) -> dict[str, str]:
    """Download only a named pinned model, atomically installing verified artifacts."""
    name = name.lower()
    spec = _model(name, manifest)
    base = Path(cache_dir) if cache_dir else default_cache_dir()
    target = base / name
    target.mkdir(parents=True, exist_ok=True)
    for artifact in spec["artifacts"].values():
        destination = target / artifact["filename"]
        if destination.is_file() and file_sha256(destination) == artifact["sha256"]:
            continue
        if not str(artifact["url"]).startswith("https://"):
            raise ModelStoreError("invalid_model_url", "Model downloads require HTTPS URLs.")
        temporary = None
        try:
            with NamedTemporaryFile(dir=target, suffix=".part", delete=False) as handle:
                temporary = Path(handle.name)
                with urlopen(Request(artifact["url"], headers={"User-Agent": "biometry-parser/0.1"}), timeout=timeout) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(chunk)
            if file_sha256(temporary) != artifact["sha256"]:
                raise ModelStoreError("model_hash_mismatch", "Downloaded artifact failed its SHA-256 integrity check.")
            os.replace(temporary, destination)
        except ModelStoreError:
            raise
        except Exception:
            raise ModelStoreError("model_download_failed", "Could not download the pinned model artifact.") from None
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
    return resolve_model(name, base, manifest=manifest)

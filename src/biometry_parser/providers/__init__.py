"""Public provider registry. Importing this module never imports ML frameworks."""
from __future__ import annotations

from importlib import metadata
from typing import Any, Callable, Protocol

from ..models import PageInput, ProviderOutput


class ProviderError(RuntimeError):
    """A safe, structured error. Messages must not contain credentials or PHI."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class Provider(Protocol):
    def describe(self) -> dict[str, Any]: ...
    def recognize(self, page: PageInput) -> ProviderOutput: ...


ProviderFactory = Callable[[dict[str, Any]], Provider]
_registry: dict[str, ProviderFactory] = {}
_plugins_loaded = False
_builtin_ids = {"native_pdf", "rapidocr", "paddleocr_api", "gemini"}
_builtin_factories: dict[str, ProviderFactory] = {}


def register_provider(provider_id: str, factory: ProviderFactory, *, replace: bool = False) -> None:
    """Register a factory accepting a profile dictionary, without private imports."""
    if not provider_id or (provider_id in _registry and not replace):
        raise ValueError(f"Provider already registered or invalid: {provider_id!r}")
    _registry[provider_id] = factory


def _load_plugins() -> None:
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for entry in metadata.entry_points(group="biometry_parser.providers"):
        if entry.name not in _registry:
            # Only installed entry points are executable; config never evaluates Python.
            _registry[entry.name] = entry.load()


def create_provider(provider_id: str, config: dict[str, Any] | None = None) -> Provider:
    _load_plugins()
    factory = _registry.get(provider_id)
    if factory is None:
        raise ProviderError("unknown_provider", "The configured provider is not registered.")
    return factory(config or {})


def registered_factory_spec(provider_id: str) -> str | None:
    """Transfer an importable custom registration to an isolated Python process."""
    _load_plugins()
    if provider_id not in _registry or _registry[provider_id] is _builtin_factories.get(provider_id):
        return None
    factory = _registry[provider_id]
    module = getattr(factory, "__module__", "")
    qualname = getattr(factory, "__qualname__", "")
    if not module or not qualname or "<" in qualname or module == "__main__":
        raise ProviderError("nonimportable_provider", "Isolated providers require an importable top-level factory or installed entry point.")
    return f"{module}:{qualname}"


def list_providers() -> list[dict[str, Any]]:
    """Static inventory: does not load models, inspect keys or perform network I/O."""
    _load_plugins()
    builtins = {
        "native_pdf": ("native_pdf", "spans", False),
        "rapidocr": ("paddle_ppocr", "spans", False),
        "paddleocr_api": ("configured_paddle_api", "spans_or_claims", True),
        "gemini": ("gemini", "claims", True),
    }
    return [dict(id=name, family=builtins.get(name, ("custom", "unknown", None))[0], kind=builtins.get(name, ("custom", "unknown", None))[1], remote=builtins.get(name, ("custom", "unknown", None))[2]) for name in sorted(_registry)]


def _local(name: str) -> ProviderFactory:
    def factory(config):
        from . import local
        return getattr(local, name)(config)
    return factory


def _vision(name: str) -> ProviderFactory:
    def factory(config):
        from . import vision
        return getattr(vision, name)(config)
    return factory


def _paddle_api(config):
    from .paddle_api import PaddleOCRAPIProvider
    return PaddleOCRAPIProvider(config)


for _name, _factory in {
    "native_pdf": _local("NativePDFProvider"),
    "rapidocr": _local("RapidOCRProvider"),
    "paddleocr_api": _paddle_api,
    "gemini": _vision("GeminiProvider"),
}.items():
    register_provider(_name, _factory)
    _builtin_factories[_name] = _factory

__all__ = ["Provider", "ProviderError", "ProviderFactory", "register_provider", "create_provider", "list_providers"]

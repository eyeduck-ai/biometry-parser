"""Small application-facing API; credentials never become serialized config."""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, Literal

from .config import ExtractionConfig, Profile, load_config
from .models import ExtractionResult


class CloudRecognitionDisabled(ValueError):
    """The caller selected a potentially remote workflow without opting in."""
    code = "cloud_recognition_disabled"


def _configuration(config, workflow=None) -> ExtractionConfig:
    if config is None or isinstance(config, (str, Path)):
        return load_config(config, workflow=workflow)
    if not isinstance(config, ExtractionConfig):
        raise TypeError("config must be an ExtractionConfig or a configuration file path")
    data = config.model_dump()
    if workflow is not None:
        data["workflow"] = workflow
    return ExtractionConfig.model_validate(data)


def extract(
    source: str | Path,
    config: ExtractionConfig | str | Path | None = None,
    *,
    workflow: str | None = None,
    allow_cloud: bool = False,
    env_overrides: dict[str, str] | None = None,
    progress: Callable[[dict], None] | None = None,
    pages: list[int] | None = None,
    page_strategy: str = "all",
) -> ExtractionResult:
    """Parse with an explicit cloud boundary and independent per-call settings.

    Selecting a cloud workflow requires ``allow_cloud=True`` even when native
    pages might satisfy it. Native-first still avoids sending readable PDF pages.
    Unknown custom provider capabilities are conservatively treated as remote.
    The opt-in is not a provider billing limit or an anonymization mechanism.
    """
    if type(allow_cloud) is not bool:
        raise TypeError("allow_cloud must be a boolean")
    selected = _configuration(config, workflow)
    recipe = selected.workflows[selected.workflow]
    if not allow_cloud and any(
        selected.profiles[name].adapter not in {"native_pdf", "rapidocr"}
        for name in recipe.profiles
    ):
        raise CloudRecognitionDisabled("Set allow_cloud=True to use the selected remote or custom provider.")
    credentials = dict(env_overrides or {})
    # Resolve only active credential variables at execution time. This also
    # makes the pipeline's output-echo guard cover environment-based keys.
    for name in recipe.profiles:
        profile = selected.profiles[name]
        variable = profile.api_key_env or {"gemini":"GEMINI_API_KEY", "paddleocr_api":"PADDLEOCR_ACCESS_TOKEN"}.get(profile.adapter)
        if variable and variable not in credentials and os.environ.get(variable):
            credentials[variable] = os.environ[variable]
    from .pipeline import extract as run
    return run(source, selected, progress=progress, env_overrides=credentials,
               pages=pages, page_strategy=page_strategy)


class BiometryParser:
    """Configure once, then parse PDF or image paths from another Python project.

    Local PP-OCRv6 small is the default fallback; usable PDF text is parsed first.
    Supplying ``config`` selects an advanced workflow without modifying its source.
    Inline API credentials live only in memory and the recognition child's env.
    """

    __slots__ = ("_config", "_allow_cloud", "_env_overrides")

    def __init__(
        self,
        backend: Literal["local", "native", "gemini", "paddle_api"] = "local",
        *,
        model: str | None = None,
        allow_cloud: bool = False,
        api_key: str | None = None,
        model_cache: str | Path | None = None,
        timeout_seconds: float = 600,
        threads: int = 2,
        native_first: bool = True,
        dpi: int = 300,
        date_order: Literal["mdy", "dmy", "reject_ambiguous"] = "mdy",
        config: ExtractionConfig | str | Path | None = None,
        workflow: str | None = None,
    ):
        if type(allow_cloud) is not bool or type(native_first) is not bool:
            raise TypeError("allow_cloud and native_first must be booleans")
        if type(threads) is not int or not 1 <= threads <= 64:
            raise ValueError("threads must be an integer between 1 and 64")
        if (not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool)
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 3600):
            raise ValueError("timeout_seconds must be a finite number between 0 and 3600")
        if backend not in {"local", "native", "gemini", "paddle_api"}:
            raise ValueError("backend must be local, native, gemini or paddle_api")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("model must be a non-empty model identifier")
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
            raise ValueError("api_key must be a non-empty string")
        if config is not None and model is not None:
            raise ValueError("Set model in the supplied config instead of combining config and model")
        if config is not None and (backend != "local" or model_cache is not None or timeout_seconds != 600
                                   or threads != 2 or not native_first or dpi != 300 or date_order != "mdy"):
            raise ValueError("Set backend, runtime and document options in the supplied config")

        if config is not None:
            selected = _configuration(config, workflow)
        else:
            mapping = {"local":"local", "native":"native_only", "gemini":"gemini", "paddle_api":"paddle_api"}
            selected_workflow = workflow or mapping[backend]
            # Use the Flash defaults for an explicitly selected non-Lite model.
            if backend == "gemini" and model == "gemini-3.6-flash" and workflow is None:
                selected_workflow = "gemini_flash"
            selected = load_config(workflow=selected_workflow)
            selected = ExtractionConfig.model_validate({**selected.model_dump(), "dpi":dpi, "date_order":date_order})
            recipe = selected.workflows[selected.workflow]
            recipe.native_first = native_first if backend != "native" else False
            for name in recipe.profiles:
                profile = selected.profiles[name]
                data = {**profile.model_dump(), "timeout_seconds":timeout_seconds}
                if model is not None:
                    data["model"] = model
                options = dict(profile.options)
                if profile.adapter == "rapidocr":
                    options["threads"] = threads
                    if model_cache is not None:
                        options["cache_dir"] = str(model_cache)
                data["options"] = options
                selected.profiles[name] = Profile.model_validate(data)

        self._env_overrides = {}
        if api_key is not None:
            recipe = selected.workflows[selected.workflow]
            variables = set()
            for name in recipe.profiles:
                profile = selected.profiles[name]
                variable = profile.api_key_env or {"gemini":"GEMINI_API_KEY", "paddleocr_api":"PADDLEOCR_ACCESS_TOKEN"}.get(profile.adapter)
                if variable:
                    variables.add(variable)
            if len(variables) != 1:
                raise ValueError("Inline api_key requires exactly one credential variable in the selected workflow")
            self._env_overrides[next(iter(variables))] = api_key
        self._config = selected
        self._allow_cloud = allow_cloud

    @property
    def config(self) -> ExtractionConfig:
        """Return a credential-free snapshot; configure advanced workflows explicitly."""
        return self._config.model_copy(deep=True)

    def parse(self, source: str | Path, *, pages: list[int] | None = None,
              page_strategy: str = "all", progress: Callable[[dict], None] | None = None) -> ExtractionResult:
        return extract(source, self._config, allow_cloud=self._allow_cloud,
                       env_overrides=self._env_overrides, progress=progress,
                       pages=pages, page_strategy=page_strategy)

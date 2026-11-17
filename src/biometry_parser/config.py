"""Validated, data-only experiment and production workflow configuration."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tomllib
from typing import Any, Literal
from pydantic import Field, model_validator
from .models import Contract

class Profile(Contract):
    adapter: str
    model: str = ""
    python: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    independence_group: str | None = None
    timeout_seconds: float = Field(default=120, gt=0, le=3600)
    api_key_env: str | None = None
    base_url: str | None = None

class Workflow(Contract):
    profiles: list[str] = Field(default_factory=list, min_length=1)
    mode: Literal["single", "crosscheck"] = "crosscheck"
    native_first: bool = False

    @model_validator(mode="after")
    def unique_profiles(self):
        if len(self.profiles) != len(set(self.profiles)):
            raise ValueError("duplicate workflow profiles are not independent evidence")
        if self.mode == "single" and len(self.profiles) != 1:
            raise ValueError("single workflows require exactly one profile")
        return self

def _default_profiles() -> dict[str, Profile]:
    return {
        "native": Profile(adapter="native_pdf"),
        "rapid_v6_small": Profile(adapter="rapidocr", model="ppocr-v6-small", timeout_seconds=600,
            options={"threads":2,"rec_batch_num":1,"text_score":0.0,"continue_on_page_error":True}),
        "gemini_lite": Profile(adapter="gemini", model="gemini-3.5-flash-lite", api_key_env="GEMINI_API_KEY", timeout_seconds=600,
            options={"max_image_side":3072,"max_tokens":32768,"retries":0,"continue_on_page_error":True}),
        "gemini_flash": Profile(adapter="gemini", model="gemini-3.6-flash", api_key_env="GEMINI_API_KEY", timeout_seconds=600,
            options={"max_image_side":3072,"max_tokens":32768,"retries":0,"omit_sampling_parameters":True,"continue_on_page_error":True}),
        "paddle_api": Profile(adapter="paddleocr_api",model="PP-OCRv6",api_key_env="PADDLEOCR_ACCESS_TOKEN",timeout_seconds=600,
            base_url="https://paddleocr.aistudio-app.com",
            options={"request_timeout_seconds":120,"poll_interval_seconds":1,"continue_on_page_error":True}),
    }


def _default_workflows() -> dict[str, Workflow]:
    return {
        "default": Workflow(profiles=["rapid_v6_small"],mode="single",native_first=True),
        "local": Workflow(profiles=["rapid_v6_small"],mode="single",native_first=True),
        "native_only": Workflow(profiles=["native"],mode="single"),
        "rapid_only": Workflow(profiles=["rapid_v6_small"],mode="single"),
        "gemini": Workflow(profiles=["gemini_lite"],mode="single",native_first=True),
        "gemini_lite": Workflow(profiles=["gemini_lite"],mode="single",native_first=True),
        "gemini_flash": Workflow(profiles=["gemini_flash"],mode="single",native_first=True),
        "gemini36": Workflow(profiles=["gemini_flash"],mode="single",native_first=True),
        "paddle_api": Workflow(profiles=["paddle_api"],mode="single",native_first=True),
    }


class ExtractionConfig(Contract):
    profiles: dict[str, Profile] = Field(default_factory=_default_profiles)
    workflows: dict[str, Workflow] = Field(default_factory=_default_workflows)
    workflow: str = "default"
    date_order: Literal["mdy", "dmy", "reject_ambiguous"] = "mdy"
    dpi: int = Field(default=300, ge=72, le=600)
    max_page_pixels: int = Field(default=20_000_000, ge=100_000)
    max_pages: int = Field(default=50, ge=1)
    cache_dir: str | None = None

    @model_validator(mode="after")
    def references(self):
        if self.workflow not in self.workflows:
            raise ValueError(f"unknown workflow: {self.workflow}")
        for name, workflow in self.workflows.items():
            for profile in workflow.profiles:
                if profile not in self.profiles:
                    raise ValueError(f"workflow {name} refers to unknown profile {profile}")
        return self

    def fingerprint(self) -> str:
        data = self.model_dump(mode="json")
        # Config contains environment VARIABLE NAMES, never resolved secret values.
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def load_config(path: str | Path | None = None, *, workflow: str | None = None) -> ExtractionConfig:
    if path is None:
        config = ExtractionConfig()
    else:
        path = Path(path)
        with path.open("rb") as handle:
            data = tomllib.load(handle) if path.suffix.lower() == ".toml" else json.load(handle)
        if workflow is not None:
            data["workflow"] = workflow
        config = ExtractionConfig.model_validate(data)
    if workflow is not None:
        config = ExtractionConfig.model_validate({**config.model_dump(), "workflow": workflow})
    return config

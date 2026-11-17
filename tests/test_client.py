"""Application API, consent boundary and per-call credential isolation."""
from pathlib import Path
import os
import subprocess
import sys

import pytest

from biometry_parser import BiometryParser, CloudRecognitionDisabled, ExtractionConfig, extract
from biometry_parser.models import ExtractionResult


def test_constructor_is_inert_and_independent_of_the_working_directory(tmp_path):
    script = """
import sys
import urllib.request
urllib.request.urlopen=lambda *a,**k: (_ for _ in ()).throw(AssertionError('network'))
from biometry_parser import BiometryParser, list_providers
p=BiometryParser()
assert p.config.profiles['rapid_v6_small'].model=='ppocr-v6-small'
assert p.config.workflows[p.config.workflow].native_first
assert {p['id'] for p in list_providers()}=={'native_pdf','rapidocr','gemini','paddleocr_api'}
assert not {'rapidocr','onnxruntime','paddle','torch','tensorflow','cv2','biometry_ocr'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)


@pytest.mark.parametrize("backend", ["gemini", "paddle_api"])
def test_cloud_requires_explicit_opt_in_before_reading_input(backend):
    parser = BiometryParser(backend=backend)
    with pytest.raises(CloudRecognitionDisabled):
        parser.parse("input-must-not-be-read.png")


def test_environment_secret_is_scoped_to_active_profile_and_not_serialized(monkeypatch):
    from biometry_parser import pipeline
    key = "synthetic-gemini-credential-not-real"
    monkeypatch.setenv("GEMINI_API_KEY", key)
    monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "synthetic-unrelated-credential")
    captured = {}
    def run(source, config, **options):
        captured.update(config=config, **options)
        return ExtractionResult(source_sha256="test", workflow=config.workflow)
    monkeypatch.setattr(pipeline, "extract", run)
    parser = BiometryParser(backend="gemini", allow_cloud=True)
    assert key not in repr(parser) and key not in parser.config.model_dump_json()
    result = parser.parse("not-read-by-fixture.png")
    assert captured["env_overrides"] == {"GEMINI_API_KEY":key}
    assert os.environ["GEMINI_API_KEY"] == key
    assert key not in result.model_dump_json()


def test_inline_key_overrides_only_child_without_mutating_environment(monkeypatch):
    from biometry_parser import pipeline
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-ambient-key")
    captured = {}
    def run(source, config, **options):
        captured.update(options)
        return ExtractionResult(source_sha256="test", workflow=config.workflow)
    monkeypatch.setattr(pipeline, "extract", run)
    parser = BiometryParser(backend="gemini", api_key="synthetic-inline-key", allow_cloud=True)
    parser.parse("fixture.png")
    assert captured["env_overrides"] == {"GEMINI_API_KEY":"synthetic-inline-key"}
    assert os.environ["GEMINI_API_KEY"] == "synthetic-ambient-key"
    assert "synthetic-inline-key" not in parser.config.model_dump_json()


def test_configuration_is_snapshotted_and_non_lite_uses_its_own_defaults():
    parser = BiometryParser(backend="gemini", model="gemini-3.6-flash", allow_cloud=True)
    config = parser.config
    selected = config.profiles[config.workflows[config.workflow].profiles[0]]
    assert selected.model == "gemini-3.6-flash"
    assert selected.options["omit_sampling_parameters"] is True
    selected.model = "changed-in-snapshot"
    assert parser.config.profiles[config.workflows[config.workflow].profiles[0]].model == "gemini-3.6-flash"


def test_custom_config_is_not_mutated_by_calls(monkeypatch):
    from biometry_parser import pipeline
    config = ExtractionConfig()
    original = config.model_dump_json()
    def run(source, selected, **options):
        selected.profiles.clear()
        return ExtractionResult(source_sha256="test", workflow=selected.workflow)
    monkeypatch.setattr(pipeline, "extract", run)
    parser = BiometryParser(config=config)
    parser.parse("fixture.png")
    parser.parse("fixture.png")
    assert parser.config.profiles and config.model_dump_json() == original


@pytest.mark.parametrize("options", [
    {"threads":True}, {"threads":0}, {"timeout_seconds":True},
    {"timeout_seconds":float("inf")}, {"timeout_seconds":0},
    {"allow_cloud":"yes"}, {"native_first":"yes"}, {"backend":"unknown"},
    {"model":""}, {"api_key":""},
])
def test_invalid_application_options_fail_fast(options):
    with pytest.raises((TypeError, ValueError)):
        BiometryParser(**options)


def test_advanced_config_rejects_silently_ignored_shortcuts():
    with pytest.raises(ValueError, match="supplied config"):
        BiometryParser(config=ExtractionConfig(), model_cache=Path("models"))
    with pytest.raises(ValueError, match="supplied config"):
        BiometryParser(config=ExtractionConfig(), model="alternative")


def test_custom_unknown_provider_requires_explicit_consent():
    from biometry_parser import Profile, Workflow
    config = ExtractionConfig(profiles={"custom":Profile(adapter="custom-plugin")},
        workflows={"custom":Workflow(profiles=["custom"],mode="single")},workflow="custom")
    with pytest.raises(CloudRecognitionDisabled):
        extract("not-read.png", config)

"""Distribution checks, also run against the wheel in clean external CI jobs."""
import configparser
from email.parser import BytesParser
import json
import os
import re
from pathlib import Path
import zipfile

import pytest


def test_wheel_is_standalone_and_has_no_platform_or_private_artifacts():
    location = os.environ.get("BIOMETRY_PARSER_TEST_WHEEL")
    if not location:
        pytest.skip("Set BIOMETRY_PARSER_TEST_WHEEL to inspect a built distribution")
    with zipfile.ZipFile(location) as wheel:
        names = wheel.namelist()
        assert not any(name.endswith((".pdf", ".onnx", "g-api.txt", "p-api.txt")) for name in names)
        assert not any(part in name for name in names for part in ("/web/", "/workbench", "/benchmark", "/evaluation", "/outputs/"))
        assert not any(name.startswith("biometry_ocr/") for name in names)
        assert all(len(wheel.read(name)) < 2_000_000 for name in names)
        manifest = json.loads(wheel.read("biometry_parser/model_manifest.json"))
        assert set(manifest["models"]) == {"ppocr-v6-small"}
        metadata_files = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_files) == 1
        dist_info = metadata_files[0].rsplit("/", 1)[0]
        metadata = BytesParser().parsebytes(wheel.read(metadata_files[0]))
        assert metadata["Name"] == "biometry-parser"
        assert all(name.startswith(("biometry_parser/", dist_info + "/")) for name in names)
        assert not any(Path(name).suffix.lower() in {".html", ".css", ".js", ".mjs", ".tsx", ".vue"} for name in names)
        entry_points = configparser.ConfigParser()
        entry_points.read_string(wheel.read(dist_info + "/entry_points.txt").decode())
        assert dict(entry_points["console_scripts"]) == {"biometry-parser": "biometry_parser.cli:main"}
        dependencies = metadata.get_all("Requires-Dist", [])
        for line in dependencies:
            if any(engine in line for engine in ("rapidocr", "onnxruntime")):
                assert 'extra == "ocr"' in line
            assert not any(engine in line.lower() for engine in ("paddlepaddle", "torch", "easyocr", "google-genai"))
            dependency = re.split(r"[\s\[<>=!~;]", line, maxsplit=1)[0].lower().replace("_", "-")
            assert dependency not in {"fastapi", "starlette", "uvicorn", "flask", "django", "streamlit", "gradio", "dash", "panel"}


def test_installed_schema_agrees_with_the_public_contract():
    from importlib.resources import files
    from biometry_parser import ExtractionResult
    recorded = json.loads(files("biometry_parser").joinpath("schemas/extraction-result.schema.json").read_text("utf-8"))
    assert recorded == ExtractionResult.model_json_schema(mode="serialization")

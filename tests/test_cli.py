"""The small CLI exports the same structured contract without a workstation."""
import json

from biometry_parser.cli import main
from biometry_parser.config import ExtractionConfig


def test_schema_export_and_minimal_provider_inventory(tmp_path, capsys):
    output=tmp_path/"result.schema.json"
    assert main(["schema","-o",str(output)])==0
    schema=json.loads(output.read_text("utf-8"))
    assert {"eyes","iol_calculations"} <= set(schema["properties"])
    assert main(["providers","list"])==0
    inventory=json.loads(capsys.readouterr().out)
    assert {row["id"] for row in inventory} >= {"native_pdf","rapidocr","gemini","paddleocr_api"}
    assert not {"easyocr","tesseract","openai_compatible","paddleocr"} & {row["id"] for row in inventory}


def test_default_configuration_is_cpu_first_and_models_are_optional():
    config=ExtractionConfig()
    assert config.workflows["default"].native_first
    assert config.workflows["default"].profiles==["rapid_v6_small"]
    assert config.profiles["rapid_v6_small"].model=="ppocr-v6-small"
    assert config.profiles["gemini_lite"].model=="gemini-3.5-flash-lite"
    assert config.profiles["gemini_flash"].model=="gemini-3.6-flash"
    assert config.profiles["paddle_api"].model=="PP-OCRv6"


def test_error_output_does_not_echo_config_data(tmp_path,capsys):
    config=tmp_path/"bad.toml"
    config.write_text('private="synthetic-sensitive-fixture"',encoding="utf-8")
    assert main(["extract","missing.pdf","-c",str(config)])==1
    assert "synthetic-sensitive-fixture" not in capsys.readouterr().err

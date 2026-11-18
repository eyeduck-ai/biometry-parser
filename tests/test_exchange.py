"""Compact exchange retains canonical meaning without shipping OCR evidence."""
import json
from decimal import Decimal

import pytest

from biometry_parser import ExtractionResult, to_exchange_document
from biometry_parser.cli import main
from biometry_parser.models import (
    Box, EYE_FIELDS, FieldResult, IOLCalculation, IOLConstantClaim, IOLRow,
    Issue, Observation, RunRecord,
)

RAW = "SYNTHETIC-RAW-EVIDENCE-" * 1000


def candidate(value, unit=None):
    return FieldResult(
        value=None, candidate_value=value, unit=unit,
        status="unverified", data_status="present",
        verification_status="unverified",
    )


def source_result():
    box = Box(x0=.1, y0=.2, x1=.3, y1=.4)
    observation = Observation(
        field="al", eye="OD", value=Decimal("23.45"), unit="mm",
        page_number=2, bbox=box, raw_text=RAW, provider_id="synthetic",
        profile_id="synthetic-profile", model_id="synthetic-model",
    )
    result = ExtractionResult(
        source_sha256="a" * 64, workflow="synthetic-test",
        patient_id=candidate("000007"), gender=candidate("female"),
        measurement_date=candidate("2026-11-18"),
        verification_state="needs_review", completeness={"present": 10},
        issues=[Issue(code="review_required", message="SYNTHETIC review")],
        metadata={
            "source_page_count": 4, "selected_pages": [2, 4],
            "source_state_counts": {"present": 10, "conflict": 1},
            "processing_status": "ok",
            "page_plan": {"raw": RAW},
            "unknown_large_metadata": RAW,
            "biometry_clip": {
                "format": "biometry-clipboard", "schemaVersion": "1.3",
                "eyes": {"OD": {"toric": True, "sia": "0.05", "target": "0"},
                         "OS": {"lvcType": "none"}},
            },
        },
        runs=[RunRecord(
            profile_id="synthetic-profile", provider_id="synthetic",
            observations=[observation], metadata={"raw": RAW},
        )],
    )
    result.eyes["OD"].al = FieldResult(
        value=Decimal("23.45"), candidate_value=Decimal("23.45"), unit="mm",
        status="verified", data_status="present",
        verification_status="cross_checked", candidates=[observation],
    )
    result.eyes["OD"].acd = candidate(Decimal("3.2"), "mm")
    for name, state in [
        ("lt", "not_measured"), ("cct", "blank"), ("wtw", "absent"),
        ("mean_k", "conflict"), ("corneal_astigmatism", "unreadable"),
        ("corneal_astigmatism_axis", "not_found"),
    ]:
        result.eyes["OD"].__setattr__(name, FieldResult(
            unit=result.eyes["OD"].scalar(name).unit,
            status=state, data_status=state,
        ))
    result.eyes["OS"].k1.axis = candidate(Decimal("0"), "degree")
    result.eyes["OS"].k2.axis = candidate(Decimal("180"), "degree")
    result.eyes["OS"].cct = candidate(Decimal("543"), "um")
    for index, (formula, source) in enumerate([("Formula A", "K"), ("Formula B", "TK")]):
        result.iol_calculations.append(IOLCalculation(
            table_id=f"synthetic-{index}", page_number=index + 2,
            eye="OD", lens_model="Same synthetic lens",
            lens_type="non_toric", formula=formula,
            keratometry_source=source, target_refraction=Decimal("0"),
            provider_id="synthetic", profile_id="synthetic-profile",
            model_id="synthetic-model", model_hash="b" * 64,
            raw_text=RAW, bbox=box, flags=["source_flag"],
            constants=[IOLConstantClaim(
                name="LF", value="Default", raw_text=RAW, bbox=box,
            )],
            rows=[IOLRow(
                iol_power=Decimal("20"), predicted_refraction=Decimal("0"),
                row_kind="candidate", selection="marked",
                selection_evidence=RAW, highlight_status="highlighted",
                highlight_evidence=RAW, raw_text=RAW, bbox=box,
                flags=["synthetic_selection"],
            )],
        ))
    return result


def final_fields(result):
    fields = {name: getattr(result, name).model_dump(mode="json", exclude={"candidates"})
              for name in ("patient_id", "gender", "measurement_date")}
    for eye in ("OD", "OS"):
        for name in EYE_FIELDS:
            fields[f"{eye}.{name}"] = result.eyes[eye].scalar(name).model_dump(
                mode="json", exclude={"candidates"})
    return fields


def test_compact_roundtrip_preserves_all_37_final_fields_and_source_meaning():
    result = source_result()
    document = to_exchange_document(result)
    loaded = ExtractionResult.model_validate_json(json.dumps(document))
    assert len(final_fields(result)) == 37
    assert final_fields(loaded) == final_fields(result)
    assert document["schema_version"] == "1.2"
    assert loaded.source_sha256 == result.source_sha256
    assert loaded.workflow == result.workflow
    assert loaded.verification_state == result.verification_state
    assert loaded.completeness == result.completeness
    assert loaded.issues == result.issues
    assert loaded.patient_id.candidate_value == "000007"
    assert loaded.eyes["OS"].k1.axis.candidate_value == 0
    assert loaded.eyes["OS"].k2.axis.candidate_value == 180
    assert loaded.eyes["OS"].cct.unit == "um"
    assert loaded.eyes["OD"].acd.value is None
    assert loaded.eyes["OD"].mean_k.data_status.value == "conflict"


def test_large_evidence_is_excluded_without_mutating_the_full_result():
    result = source_result()
    before = result.model_dump_json()
    document = to_exchange_document(result)
    encoded = json.dumps(document)
    assert "runs" not in document
    assert '"candidates":' not in encoded
    assert '"bbox":' not in encoded
    assert '"raw_text":' not in encoded
    assert RAW not in encoded
    assert len(encoded) < len(before) / 5
    assert result.model_dump_json() == before


def test_iol_tables_keep_distinct_formula_numeric_selection_and_identity_context():
    result = source_result()
    document = to_exchange_document(result)
    loaded = ExtractionResult.model_validate(document)
    assert len(loaded.iol_calculations) == 2
    for original, table in zip(result.iol_calculations, loaded.iol_calculations):
        assert table.table_id == original.table_id
        assert table.formula == original.formula
        assert table.keratometry_source == original.keratometry_source
        assert table.page_number == original.page_number
        assert table.target_refraction == 0
        assert table.model_hash == original.model_hash
        assert table.verification_status == "unverified"
        assert table.constants[0].value == "Default"
        row = table.rows[0]
        assert row.iol_power == 20 and row.predicted_refraction == 0
        assert row.selection == "marked"
        assert row.highlight_status == "highlighted"
        assert row.flags == ["synthetic_selection"]
        assert row.selection_evidence == "" and row.bbox is None
    assert document["metadata"]["biometry_parser_exchange"]["profile"] == "compact"
    assert "selection_evidence" in document["metadata"]["biometry_parser_exchange"]["omitted"]


def test_metadata_keeps_portable_clip_settings_and_context_as_an_independent_copy():
    result = source_result()
    document = to_exchange_document(result)
    metadata = document["metadata"]
    assert metadata["biometry_clip"] == result.metadata["biometry_clip"]
    assert metadata["source_page_count"] == 4
    assert metadata["selected_pages"] == [2, 4]
    assert metadata["processing_status"] == "ok"
    assert "page_plan" not in metadata and "unknown_large_metadata" not in metadata
    metadata["biometry_clip"]["eyes"]["OD"]["sia"] = "0.10"
    assert result.metadata["biometry_clip"]["eyes"]["OD"]["sia"] == "0.05"


def test_compact_export_requires_a_current_validated_result_and_never_invents_observations():
    with pytest.raises(TypeError):
        to_exchange_document({})
    legacy = ExtractionResult(schema_version="1.0", source_sha256="test", workflow="test")
    with pytest.raises(ValueError, match="schema 1.2"):
        to_exchange_document(legacy)
    result = ExtractionResult(source_sha256="c" * 64, workflow="manual-import")
    result.eyes["OD"].al = candidate(Decimal("23.45"), "mm")
    document = to_exchange_document(result)
    loaded = ExtractionResult.model_validate(document)
    assert document["eyes"]["OD"]["al"]["value"] is None
    assert document["eyes"]["OD"]["al"]["candidate_value"] == 23.45
    assert loaded.eyes["OD"].al.candidates == [] and loaded.runs == []
    assert loaded.eyes["OS"].al.data_status.value == "not_found"


def test_cli_compact_is_opt_in_and_preserves_existing_exit_status(monkeypatch, tmp_path):
    result = source_result()
    monkeypatch.setattr("biometry_parser.client.extract", lambda *args, **kwargs: result)
    normal, compact = tmp_path / "normal.json", tmp_path / "compact.json"
    assert main(["extract", "synthetic.pdf", "-o", str(normal)]) == 0
    assert main(["extract", "synthetic.pdf", "--compact", "-o", str(compact)]) == 0
    full = json.loads(normal.read_text("utf-8"))
    small = json.loads(compact.read_text("utf-8"))
    assert full["runs"] and full["eyes"]["OD"]["al"]["candidates"]
    assert "biometry_parser_exchange" not in full["metadata"]
    assert "runs" not in small and "candidates" not in small["eyes"]["OD"]["al"]
    assert final_fields(ExtractionResult.model_validate(small)) == final_fields(result)
    result.runs[0].outcome = "partial"
    assert main(["extract", "synthetic.pdf", "--compact", "-o", str(compact)]) == 2

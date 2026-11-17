from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from PIL import Image
import pytest

from biometry_parser.pipeline import extract
from biometry_parser.config import ExtractionConfig, Profile, Workflow
from biometry_parser.documents import native_text_usable
from biometry_parser.models import Box, FieldClaim, IOLCalculation, IOLRow, Issue, PageInput, ParsedPage, ProviderOutput, Span
from biometry_parser.report_templates import ReportTemplate, inspect_report, list_report_templates, register_report_template


def native_spans(*, device="700", reversed_eyes=False, broken_values=False):
    def span(text, x, y, right):
        return Span(text=text, bbox=Box(x0=x, y0=y, x1=right, y1=y+.015))
    od, os = ("OS", "OD") if reversed_eyes else ("OD", "OS")
    return [span("ZEISS IOLMaster " + device, .1, .04, .8),
            span("Patient ID: DEMO-001", .1, .10, .6),
            span("Date of measurement: 2026-11-15", .1, .14, .8),
            span(od, .1, .25, .2), span(os, .7, .25, .8),
            span("Biometric values", .1, .32, .5),
            span("AL: unreadable mm" if broken_values else "AL: 23.62 mm", .1, .4, .4),
            span("AL: unreadable mm" if broken_values else "AL: 23.41 mm", .6, .4, .9),
            span("ACD: unreadable mm" if broken_values else "ACD: 3.21 mm", .1, .45, .4),
            span("ACD: unreadable mm" if broken_values else "ACD: 3.39 mm", .6, .45, .9)]


def test_template_registry_only_claims_registered_parser_support_and_observed_eye_order():
    assert any(item["id"] == "zeiss_iolmaster_700" and item["supported"] for item in list_report_templates())
    normal = inspect_report(native_spans())
    reversed_report = inspect_report(native_spans(reversed_eyes=True))
    assert normal["layout"]["eye_order"] == "OD_left_OS_right"
    assert reversed_report["layout"]["eye_order"] == "OS_left_OD_right"
    assert normal["layout"]["validated"] and reversed_report["layout"]["validated"]
    assert normal["layout"]["automatic_crop_applied"] is False
    unknown = inspect_report(native_spans(device="500"))
    assert unknown["template_id"] is None and not unknown["supported"]
    assert unknown["device"] == "IOLMaster 500" and unknown["explicit_unsupported_device"]
    page = PageInput(page_number=1, image_path="not-rendered", width=100, height=140, native_spans=native_spans(device="500"))
    assert not native_text_usable(page)


def test_template_matcher_without_parser_does_not_enable_a_new_manufacturer(monkeypatch):
    import biometry_parser.report_templates as templates
    monkeypatch.setattr(templates, "_TEMPLATES", dict(templates._TEMPLATES))
    # A matcher is just identification; this deliberately registers no parser.
    register_report_template(ReportTemplate("routing_test_without_parser", "Synthetic", "DEMO-DEVICE",
                                           lambda text: "DEMO-DEVICE" in text))
    spans = native_spans()
    spans[0] = spans[0].model_copy(update={"text":"DEMO-DEVICE"})
    result = inspect_report(spans)
    assert result["template_id"] == "routing_test_without_parser"
    assert not result["supported"] and not result["native_quality"]["usable"]
    with pytest.raises(ValueError):
        register_report_template(ReportTemplate("routing_test_without_parser", "Synthetic", "DEMO-DEVICE", lambda text: False))


def test_explicit_other_device_name_does_not_get_reused_as_zeiss_layout():
    spans = native_spans()
    spans[0] = spans[0].model_copy(update={"text":"LENSTAR LS 900"})
    inspection = inspect_report(spans)
    assert inspection["device"] == "LENSTAR" and inspection["explicit_unsupported_device"]
    assert not inspection["supported"] and not inspection["native_quality"]["usable"]


def test_calculation_table_anchor_identifies_intent_without_requiring_one_exact_title():
    spans = native_spans()
    spans.extend([Span(text="IOL (D)", bbox=Box(x0=.1,y0=.55,x1=.18,y1=.57)),
                  Span(text="Ref. (D)", bbox=Box(x0=.2,y0=.55,x1=.3,y1=.57))])
    inspection = inspect_report(spans)
    assert inspection["page_kind"] == "iol_calculation"
    assert inspection["layout"]["calculation_table_anchors_validated"]


def run_with_pages(tmp_path, monkeypatch, spans_by_page, *, native_first=True, adapter="synthetic-ocr"):
    from biometry_parser import documents, worker_client
    source = tmp_path / "synthetic.pdf"
    source.write_bytes(b"synthetic-source-not-a-real-pdf")
    directory = tmp_path / "rasters"
    directory.mkdir()
    pages = [PageInput(page_number=index+1, image_path=str(directory / f"{index+1}.png"), width=100, height=140,
                       native_spans=spans, source_path=str(source)) for index, spans in enumerate(spans_by_page)]
    rendered, called = [], []
    @contextmanager
    def open_pages(path, **kwargs):
        assert kwargs["render_pages"] is False
        yield iter(pages)
    def ensure(page, **kwargs):
        if not Path(page.image_path).is_file():
            rendered.append(page.page_number)
            Image.new("RGB", (100, 140), "white").save(page.image_path)
        return page
    class Recognizer:
        startup_seconds, peak_rss_bytes, cpu_seconds = .01, 1000, .02
        last_resources = {"cpu_seconds":.02,"peak_rss_bytes":1000}
        def __init__(self, *args, **kwargs): pass
        def start(self): return {"family":"synthetic-ocr","model_id":"synthetic-v1"}
        def close(self): pass
        def recognize(self, page):
            assert not page.native_spans and Path(page.image_path).is_file()
            called.append(page.page_number)
            return ProviderOutput(kind="claims", supported=True, claims=[
                FieldClaim(field="patient_id", raw_value="DEMO-001"),
                FieldClaim(field="measurement_date", raw_value="2026-11-15"),
                FieldClaim(field="al", eye="OD", raw_value="23.62", unit="mm")])
    monkeypatch.setattr(documents, "open_document", open_pages)
    monkeypatch.setattr(documents, "ensure_page_image", ensure)
    monkeypatch.setattr(worker_client, "WorkerClient", Recognizer)
    config = ExtractionConfig(profiles={"ocr":Profile(adapter=adapter,model="synthetic-v1")},
        workflows={"test":Workflow(profiles=["ocr"],mode="single",native_first=native_first)},workflow="test")
    return extract(source, config), rendered, called


def test_mixed_document_reads_native_page_and_falls_back_only_for_scanned_page(tmp_path, monkeypatch):
    result, rendered, called = run_with_pages(tmp_path, monkeypatch, [native_spans(), []])
    assert rendered == called == [2]
    decisions = result.runs[0].metadata["routing_decisions"]
    assert [entry["route"] for entry in decisions] == ["native_pdf", "synthetic-ocr"]
    assert decisions[0]["native_parse_supported"] and not decisions[0]["fallback_to_recognizer"]
    assert decisions[1]["fallback_to_recognizer"] and "native_text_unusable" in decisions[1]["reason_codes"]
    assert result.metadata["rasterized_pages"] == [2]
    assert result.runs[0].metadata["routed_pages"] == {"native_pdf":[1],"synthetic-ocr":[2]}


def test_usable_native_page_creates_no_image_and_never_loads_a_recognizer(tmp_path, monkeypatch):
    result, rendered, called = run_with_pages(tmp_path, monkeypatch, [native_spans()])
    assert rendered == called == []
    assert result.metadata["rasterized_pages"] == [] and result.metadata["raster_render_seconds"] == 0
    assert not (tmp_path / "rasters/1.png").exists()
    assert result.runs[0].provider_id == "native_pdf" and result.runs[0].model_id == "pdfium-text"


def test_native_parser_unsupported_result_falls_back_instead_of_stopping(tmp_path, monkeypatch):
    import biometry_parser.report_templates as templates
    original = templates.parse_registered_page
    def reject_native(page, output, **kwargs):
        return ParsedPage(supported=False) if kwargs.get("model_id") == "pdfium-text" else original(page, output, **kwargs)
    monkeypatch.setattr(templates, "parse_registered_page", reject_native)
    result, rendered, called = run_with_pages(tmp_path, monkeypatch, [native_spans()])
    assert rendered == called == [1]
    decision = result.runs[0].metadata["routing_decisions"][0]
    assert decision["native_parse_supported"] is False and decision["fallback_to_recognizer"]
    assert result.eyes["OD"].al.candidate_value == Decimal("23.62")


def test_native_candidate_that_parses_no_measurements_really_falls_back(tmp_path, monkeypatch):
    result, rendered, called = run_with_pages(tmp_path, monkeypatch, [native_spans(broken_values=True)])
    assert rendered == called == [1]
    decision = result.runs[0].metadata["routing_decisions"][0]
    assert decision["native_attempted"] and decision["native_parse_supported"]
    assert decision["fallback_to_recognizer"] and "native_parse_requires_fallback" in decision["reason_codes"]
    assert result.eyes["OD"].al.candidate_value == Decimal("23.62")


def test_one_damaged_native_value_does_not_hide_behind_other_good_fields(tmp_path, monkeypatch):
    spans = native_spans()
    index = next(i for i, span in enumerate(spans) if span.text == "AL: 23.62 mm")
    spans[index] = spans[index].model_copy(update={"text":"AL: 23,62 mm"})
    result, rendered, called = run_with_pages(tmp_path, monkeypatch, [spans])
    assert rendered == called == [1]
    decision = result.runs[0].metadata["routing_decisions"][0]
    assert decision["native_parse_supported"] and decision["fallback_to_recognizer"]


def test_calculation_page_with_good_biometry_but_no_iol_rows_falls_back(tmp_path, monkeypatch):
    spans = native_spans()
    index = next(i for i, span in enumerate(spans) if span.text == "Biometric values")
    spans[index] = spans[index].model_copy(update={"text":"IOL calculation"})
    result, rendered, called = run_with_pages(tmp_path, monkeypatch, [spans])
    assert rendered == called == [1]
    decision = result.runs[0].metadata["routing_decisions"][0]
    assert decision["native_summary_observation_count"] >= 4 and decision["native_iol_table_count"] == 0
    assert "native_calculation_rows_missing" in decision["reason_codes"] and decision["fallback_to_recognizer"]


def test_explicit_native_only_does_not_silently_start_ocr_for_missing_iol_rows(tmp_path, monkeypatch):
    spans = native_spans()
    index = next(i for i, span in enumerate(spans) if span.text == "Biometric values")
    spans[index] = spans[index].model_copy(update={"text":"IOL calculation"})
    result, rendered, called = run_with_pages(tmp_path, monkeypatch, [spans], adapter="native_pdf")
    assert rendered == called == []
    decision = result.runs[0].metadata["routing_decisions"][0]
    assert decision["route"] == "native_pdf" and not decision["fallback_to_recognizer"]
    assert "native_calculation_rows_missing" in decision["reason_codes"]


def test_invalid_iol_cells_trigger_native_fallback_even_with_readable_power_and_biometry(tmp_path, monkeypatch):
    import biometry_parser.report_templates as templates
    original = templates.parse_registered_page
    def damaged_table(page, output, **kwargs):
        parsed = original(page, output, **kwargs)
        if kwargs.get("model_id") == "pdfium-text":
            parsed.iol_calculations = [IOLCalculation(table_id="synthetic", eye="OD", page_number=1,
                rows=[IOLRow(iol_power=Decimal("20.0"), predicted_refraction=Decimal("-.25")),
                      IOLRow(iol_power=None, flags=["iol_power_unreadable"])])]
            parsed.issues.append(Issue(code="invalid_iol_cell",message="synthetic damaged source cell"))
        return parsed
    monkeypatch.setattr(templates, "parse_registered_page", damaged_table)
    result, rendered, called = run_with_pages(tmp_path, monkeypatch, [native_spans()])
    assert rendered == called == [1]
    decision = result.runs[0].metadata["routing_decisions"][0]
    assert decision["native_iol_table_count"] == 1 and decision["native_iol_readable_power_row_count"] == 1
    assert decision["fallback_to_recognizer"]


def test_forced_model_comparison_does_not_take_native_shortcut(tmp_path, monkeypatch):
    result, rendered, called = run_with_pages(tmp_path, monkeypatch, [native_spans()], native_first=False)
    assert rendered == called == [1]
    decision = result.runs[0].metadata["routing_decisions"][0]
    assert decision["forced_ocr"] and not decision["native_attempted"]
    assert decision["reason_codes"] == ["forced_model_recognition"]


def test_explicit_other_device_is_not_accepted_even_if_a_provider_declares_support(tmp_path, monkeypatch):
    result, rendered, called = run_with_pages(tmp_path, monkeypatch, [native_spans(device="500")])
    assert rendered == called == [1]
    decision = result.runs[0].metadata["routing_decisions"][0]
    assert "unsupported_report_device" in decision["reason_codes"]
    assert not result.runs[0].observations and result.eyes["OD"].al.candidate_value is None

from pathlib import Path
import hashlib

from PIL import Image
import pytest

from biometry_parser.page_selection import plan_pages, select_native_coverage
from biometry_parser.models import Observation, ParsedPage


def frame_source(tmp_path):
    first = Image.new("RGB", (100, 140), "white")
    changed = first.copy()
    changed.putpixel((40, 50), (0, 0, 0))
    path = tmp_path / "pages.tiff"
    first.save(path, save_all=True, append_images=[first.copy(), changed])
    first.close()
    changed.close()
    return path


def test_auto_only_removes_exact_input_pixels_and_records_original_page(tmp_path):
    source = frame_source(tmp_path)
    plan = plan_pages(source, strategy="auto")
    assert plan["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert plan["selected_pages"] == [1, 3]
    assert plan["skipped_pages"] == [2]
    assert plan["page_details"][1]["duplicate_of"] == 1
    assert plan["page_details"][1]["reason"] == "identical_input_pixels"
    assert plan["page_details"][2]["selected"] is True  # one pixel is enough to retain
    assert plan["recognition_calls_made"] == 0 and plan["inference_used"] is False


def test_all_strategy_keeps_duplicate_pages_and_custom_preserves_original_numbers(tmp_path):
    path = frame_source(tmp_path)
    assert plan_pages(path, strategy="all")["selected_pages"] == [1, 2, 3]
    plan = plan_pages(path, strategy="custom", pages=[3, 1])
    assert plan["selected_pages"] == [1, 3]
    assert plan["source_page_count"] == 3
    assert plan_pages(path, strategy="current", pages=[2])["selected_pages"] == [2]


def test_native_coverage_keeps_values_and_missing_state_disagreements():
    present = ("al", "OD", "present", "23.62", "mm")
    different = ("al", "OD", "present", "23.63", "mm")
    absent = ("al", "OD", "not_reported", None, "mm")
    assert select_native_coverage({1: {present}, 2: {present, different}, 3: {different}}) == [2]
    assert select_native_coverage({1: {present}, 2: {different}, 3: {absent}}) == [1, 2, 3]
    assert select_native_coverage({1: {present}, 2: {present}}) == [1]


def test_uncertain_native_evidence_is_retained_even_when_both_pages_have_null_values(tmp_path, monkeypatch):
    import biometry_parser.page_selection as selection
    import pypdfium2 as pdfium
    source = tmp_path / "uncertain.pdf"
    source.write_bytes(b"synthetic")
    class Page:
        def get_size(self): return 100, 140
        def close(self): pass
        def render(self, **kwargs): pytest.fail("native-only planning must not render")
    class Document(list):
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(pdfium, "PdfDocument", lambda path: Document([Page(), Page()]))
    monkeypatch.setattr(selection, "_native_spans", lambda page: [])
    monkeypatch.setattr(selection, "native_text_usable", lambda page: True)
    def parse(page, *args, **kwargs):
        return ParsedPage(supported=True, observations=[
            Observation(field="patient_id", value="DEMO", page_number=page.page_number),
            Observation(field="measurement_date", value="2026-11-14", page_number=page.page_number),
            Observation(field="al", eye="OD", value=None, status="unreadable", page_number=page.page_number)])
    monkeypatch.setattr(selection, "parse_page", parse)
    plan = plan_pages(source)
    assert plan["selected_pages"] == [1, 2]
    assert all(detail["reason"] == "native_uncertain_evidence_requires_review" for detail in plan["page_details"])


@pytest.mark.parametrize("strategy,pages", [("custom", []), ("custom", [0]), ("custom", [True]),
    ("custom", [1, 1]), ("custom", [4]), ("current", [1, 2]), ("auto", [1]), ("all", [1]), ("bad", None)])
def test_invalid_page_plans_are_rejected(tmp_path, strategy, pages):
    with pytest.raises(ValueError):
        plan_pages(frame_source(tmp_path), strategy=strategy, pages=pages)



"""Reviewed input identity and integrity gates fail closed, without an API."""
import copy
import hashlib
import json

from PIL import Image
import pytest

from biometry_parser.input_transforms import reviewed_page
from biometry_parser.models import Box, PageInput, Span
from biometry_parser.providers import ProviderError


@pytest.fixture
def reviewed(tmp_path):
    directory = tmp_path / "study"
    directory.mkdir()
    sanitized = directory / "page-3.png"
    Image.new("RGB", (20, 30), "white").save(sanitized)
    source_hash = "a" * 64
    page = PageInput(page_number=3, image_path=str(tmp_path / "original.png"), width=20, height=30,
        native_spans=[Span(text="synthetic private text", bbox=Box(x0=0,y0=0,x1=.5,y1=.1))])
    entry = {"page_number": 3, "image_path": sanitized.name,
        "image_sha256": hashlib.sha256(sanitized.read_bytes()).hexdigest(), "width": 20, "height": 30,
        "reviewed": True, "masks": [{"x0":0,"y0":0,"x1":1,"y1":.25}],
        "header_added": {"text":"ZEISS IOLMaster 700", "contains_answers":False,
                         "origin":"synthetic_brand_context"}}
    manifest = {"kind":"biometry-ocr-sanitized-inputs", "version":"1.0", "cases":[{
        "case_id":"fixture", "source_sha256":source_hash, "source_page_count":8,
        "excluded_fields":["patient_id","gender","measurement_date"], "pages":[entry]}]}
    manifest_path = directory / "manifest.json"
    def save():
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return str(manifest_path)
    return page, source_hash, manifest, entry, sanitized, save


def reject(page, source_hash, save):
    with pytest.raises(ProviderError) as error:
        reviewed_page(page, source_hash, save())
    assert error.value.code == "input_transform_invalid"
    assert "synthetic private text" not in str(error.value)


def test_reviewed_copy_keeps_original_page_and_records_synthetic_header(reviewed):
    page, source_hash, manifest, entry, sanitized, save = reviewed
    copy_page, metadata = reviewed_page(page, source_hash, save())
    assert copy_page.page_number == 3 and copy_page.image_path == str(sanitized.resolve())
    assert copy_page.native_spans == [] and page.native_spans
    assert metadata["source_sha256"] == source_hash and metadata["page_number"] == 3
    assert metadata["image_sha256"] == entry["image_sha256"]
    assert metadata["excluded_fields"] == manifest["cases"][0]["excluded_fields"]
    assert metadata["synthetic_header"] is True
    assert metadata["header_added"] == entry["header_added"]
    assert metadata["manifest_sha256"] == hashlib.sha256((sanitized.parent / "manifest.json").read_bytes()).hexdigest()


def test_wrong_original_source_hash_cannot_select_another_case(reviewed):
    page, source_hash, manifest, entry, sanitized, save = reviewed
    reject(page, "b" * 64, save)


def test_wrong_original_page_cannot_borrow_a_reviewed_other_page(reviewed):
    page, source_hash, manifest, entry, sanitized, save = reviewed
    reject(page.model_copy(update={"page_number":4}), source_hash, save)


@pytest.mark.parametrize("review_flag", [False, None, "true", 1])
def test_missing_or_untrusted_review_marker_is_rejected(reviewed, review_flag):
    page, source_hash, manifest, entry, sanitized, save = reviewed
    entry["reviewed"] = review_flag
    reject(page, source_hash, save)


def test_modified_pixels_cannot_reuse_the_recorded_image_hash(reviewed):
    page, source_hash, manifest, entry, sanitized, save = reviewed
    Image.new("RGB", (20, 30), "black").save(sanitized)
    reject(page, source_hash, save)


@pytest.mark.parametrize("target", ["source", "manifest"])
def test_dimensions_must_match_both_source_page_and_review_manifest(reviewed, target):
    page, source_hash, manifest, entry, sanitized, save = reviewed
    if target == "source": page = page.model_copy(update={"width":21})
    else: entry["height"] = 31
    reject(page, source_hash, save)


def test_reviewed_image_cannot_escape_manifest_directory_even_with_valid_hash(reviewed):
    page, source_hash, manifest, entry, sanitized, save = reviewed
    outside = sanitized.parent.parent / "outside.png"
    outside.write_bytes(sanitized.read_bytes())
    entry["image_path"] = "../outside.png"
    reject(page, source_hash, save)


@pytest.mark.parametrize("duplicate", ["source_case", "page"])
def test_ambiguous_duplicate_source_or_page_does_not_pick_the_first_entry(reviewed, duplicate):
    page, source_hash, manifest, entry, sanitized, save = reviewed
    if duplicate == "source_case": manifest["cases"].append(copy.deepcopy(manifest["cases"][0]))
    else: manifest["cases"][0]["pages"].append(copy.deepcopy(entry))
    reject(page, source_hash, save)


@pytest.mark.parametrize("count", [0, 2, True, "8"])
def test_manifest_page_scope_must_be_valid_and_include_the_original_page(reviewed, count):
    page, source_hash, manifest, entry, sanitized, save = reviewed
    manifest["cases"][0]["source_page_count"] = count
    reject(page, source_hash, save)


def test_float_page_number_cannot_alias_the_integer_source_page(reviewed):
    page, source_hash, manifest, entry, sanitized, save = reviewed
    entry["page_number"] = 3.0
    reject(page, source_hash, save)


def test_reviewed_pixels_do_not_allow_hidden_png_text_metadata(reviewed):
    from PIL.PngImagePlugin import PngInfo
    page, source_hash, manifest, entry, sanitized, save = reviewed
    info=PngInfo();info.add_text('Comment','synthetic identifying metadata')
    Image.new('RGB',(20,30),'white').save(sanitized,pnginfo=info)
    entry['image_sha256']=hashlib.sha256(sanitized.read_bytes()).hexdigest()
    reject(page,source_hash,save)

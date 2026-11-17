import json
from urllib.error import HTTPError

import pytest
from PIL import Image

from biometry_parser.models import Box, PageInput, Span
from biometry_parser.providers import ProviderError, create_provider, registered_factory_spec
from biometry_parser.providers import paddle_api


@pytest.fixture
def page(tmp_path):
    path = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(path)
    return PageInput(page_number=6, image_path=str(path), width=100, height=100,
                     native_spans=[Span(text="DO NOT SEND PRIMARY ANSWERS", bbox=Box(x0=0, y0=0, x1=1, y1=1))])


def output_document(model="PP-OCRv6", pruned=None):
    key = "ocrResults" if model == "PP-OCRv6" else "layoutParsingResults"
    return {"result": {key: [{"prunedResult": pruned or {"rec_texts": ["AL: 23.62 mm"],
                        "rec_boxes": [[10, 20, 40, 30]], "rec_scores": [.91]}}]}}


def service(monkeypatch, document, states=None):
    calls = []
    states = iter(states or [{"state": "done", "resultUrl": {"jsonUrl": "https://results.bcebos.com/private/result.jsonl?signature=opaque"}}])
    def request(url, **kwargs):
        calls.append((url, kwargs))
        if kwargs.get("method") == "POST":
            return json.dumps({"code": 0, "data": {"jobId": "opaque-job-1"}}).encode()
        if "/api/v2/ocr/jobs/" in url:
            return json.dumps({"code": 0, "data": next(states)}).encode()
        return (json.dumps(document) + "\n").encode()
    monkeypatch.setattr(paddle_api, "_request_bytes", request)
    monkeypatch.setattr(paddle_api.time, "sleep", lambda _: None)
    monkeypatch.setenv("PADDLEOCR_ACCESS_TOKEN", "unit-test-access-token")
    return calls


def test_official_paddle_jobs_upload_poll_and_result_without_auth(page, monkeypatch):
    calls = service(monkeypatch, output_document(), states=[{"state": "pending"},
                    {"state": "done", "resultUrl": {"jsonUrl": "https://results.bcebos.com/r.jsonl?signature=opaque"}}])
    adapter = create_provider("paddleocr_api", {"model": "PP-OCRv6"})
    result = adapter.recognize(page)
    assert registered_factory_spec("paddleocr_api") is None
    assert calls[0][0] == "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    assert calls[0][1]["headers"]["Authorization"] == "Bearer unit-test-access-token"
    assert calls[0][1]["headers"]["Content-Type"].startswith("multipart/form-data; boundary=")
    assert b'name="file"; filename="page.png"' in calls[0][1]["body"]
    assert b'PP-OCRv6' in calls[0][1]["body"]
    assert b'DO NOT SEND PRIMARY ANSWERS' not in calls[0][1]["body"]
    assert b'useDocOrientationClassify' in calls[0][1]["body"]
    assert calls[-1][1]["headers"] == {}
    assert result.spans[0].bbox == Box(x0=.1, y0=.2, x1=.4, y1=.3)
    assert result.spans[0].confidence == .91
    assert result.metadata["family"] == "paddle_ppocr"
    assert result.metadata["usage_complete"] is False
    assert result.metadata["api_poll_count"] == 2
    assert "unit-test-access-token" not in result.model_dump_json()
    assert "signature=opaque" not in result.model_dump_json()


def test_paddle_requires_its_own_token_and_does_not_import_framework(monkeypatch):
    monkeypatch.delenv("PADDLEOCR_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "other-provider-token")
    with pytest.raises(ProviderError) as error:
        create_provider("paddleocr_api", {"model": "PP-OCRv6"})
    assert error.value.code == "credentials_unavailable"


@pytest.mark.parametrize("url", ["http://results.bcebos.com/x", "https://evil.example/x",
                               "https://127.0.0.1/x", "https://results.bcebos.com:8443/x",
                               "https://user:secret@results.bcebos.com/x", "https://bcebos.com.evil.example/x"])
def test_untrusted_result_url_is_rejected_before_download(page, monkeypatch, url):
    calls = service(monkeypatch, output_document(), states=[{"state": "done", "resultUrl": {"jsonUrl": url}}])
    with pytest.raises(ProviderError) as error:
        create_provider("paddleocr_api").recognize(page)
    assert error.value.code == "invalid_provider_response"
    assert len(calls) == 2


def test_custom_result_storage_requires_explicit_allowlist(page, monkeypatch):
    calls = service(monkeypatch, output_document(), states=[{"state": "done", "resultUrl": {"jsonUrl": "https://storage.my-proxy.example/result"}}])
    result = create_provider("paddleocr_api", {"options": {"result_host_allowlist": ["storage.my-proxy.example"]}}).recognize(page)
    assert result.spans and calls[-1][1]["headers"] == {}


def test_failed_job_message_does_not_leak(monkeypatch, page):
    service(monkeypatch, output_document(), states=[{"state": "failed", "errorMsg": "unit-test-access-token private patient text"}])
    with pytest.raises(ProviderError) as error:
        create_provider("paddleocr_api").recognize(page)
    assert error.value.code == "api_job_failed"
    assert "patient" not in str(error.value) and "unit-test" not in str(error.value)


def test_http_errors_do_not_leak(monkeypatch):
    def fail(*args, **kwargs):
        raise HTTPError("https://service/?token=secret", 401, "secret patient text", None, None)
    monkeypatch.setattr(paddle_api, "urlopen", fail)
    with pytest.raises(ProviderError) as error:
        paddle_api._request_bytes("https://service")
    assert str(error.value) == "PaddleOCR API returned HTTP 401."


def test_http_transport_bounds_response_and_uses_deadline(monkeypatch):
    captured = {}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self, limit):
            captured["limit"] = limit
            return b'{"code":0,"data":{}}'
    def opened(request, **kwargs):
        captured.update(request=request, **kwargs)
        return Response()
    monkeypatch.setattr(paddle_api, "urlopen", opened)
    monkeypatch.setattr(paddle_api.time, "monotonic", lambda: 10)
    raw = paddle_api._request_bytes("https://service", method="POST", body=b'body', headers={"Authorization": "Bearer unit-test-access-token"}, timeout=30, deadline=12)
    assert json.loads(raw)["code"] == 0
    assert captured["timeout"] == 2
    assert captured["limit"] == 8 * 1024 * 1024 + 1
    assert captured["request"].data == b'body'


def test_pending_job_has_total_deadline_and_is_never_resubmitted(page, monkeypatch):
    calls = service(monkeypatch, output_document(), states=[{"state": "pending"}])
    ticks = iter([0, 1])
    monkeypatch.setattr(paddle_api.time, "monotonic", lambda: next(ticks))
    with pytest.raises(ProviderError) as error:
        create_provider("paddleocr_api", {"timeout_seconds": 1, "options": {"request_timeout_seconds": .5}}).recognize(page)
    assert error.value.code == "provider_timeout"
    assert [c[1].get("method") for c in calls].count("POST") == 1


@pytest.mark.parametrize("pruned", [
    {"rec_texts": ["one", "two"], "rec_boxes": [[0, 0, 20, 20]]},
    {"rec_texts": ["one"], "rec_boxes": [[0, 0, 200, 20]]},
    {"rec_texts": ["one"], "rec_boxes": [[0, 0, 20, 20]], "rec_scores": [float("nan")]},
])
def test_invalid_ocr_arrays_geometry_and_confidence_fail_closed(page, monkeypatch, pruned):
    service(monkeypatch, output_document(pruned=pruned))
    with pytest.raises(ProviderError) as error:
        create_provider("paddleocr_api").recognize(page)
    assert error.value.code == "invalid_provider_response"


def test_vl_explicit_eyes_and_dash_states_with_only_real_block_bbox(page, monkeypatch):
    pruned = {"parsing_res_list": [
        {"block_content": "ZEISS IOLMaster 700\nPatient ID: 00123\nGender: Male\nDate of measurement: 9/11/2026", "block_bbox": [0, 0, 100, 15]},
        {"block_content": "<table><tr><th>Field</th><th>OD</th><th>OS</th></tr><tr><td>AL (mm)</td><td>23.62</td><td>23.41</td></tr><tr><td>K1 (D)</td><td>44.91 D @ 165°</td><td>45.21 D @ 71°</td></tr><tr><td>TK1 (D)</td><td>---</td><td>---</td></tr></table>", "block_bbox": [0, 20, 100, 80]},
    ]}
    service(monkeypatch, output_document("PaddleOCR-VL-1.6", pruned))
    result = create_provider("paddleocr_api", {"model": "PaddleOCR-VL-1.6"}).recognize(page)
    assert result.kind == "claims" and result.supported
    values = {(c.field, c.eye): c for c in result.claims}
    assert values["al", "OD"].raw_value == "23.62"
    assert values["al", "OS"].raw_value == "23.41"
    assert values["al", "OD"].unit == "mm"
    assert values["k1_axis", "OD"].raw_value == "165"
    assert values["tk1_power", "OS"].missing == "not_measured"
    assert values["tk1_axis", "OS"].missing == "not_measured"
    assert values["patient_id", None].raw_value == "00123"
    assert values["al", "OD"].bbox == values["al", "OS"].bbox == Box(x0=0, y0=.2, x1=1, y1=.8)
    assert result.metadata["evidence_precision"] == "block_or_page"
    assert result.metadata["family"] == "paddle_ocr_vl"


def test_vl_markdown_eyes_as_rows_and_ambiguous_layout_is_not_guessed(page, monkeypatch):
    document = output_document("PaddleOCR-VL-1.6", {"parsing_res_list": []})
    document["result"]["layoutParsingResults"][0]["markdown"] = {"text": "ZEISS IOLMaster 700\n| Eye | AL (mm) |\n| --- | --- |\n| OD | 23.62 |\n| OS | 23.41 |\n"}
    service(monkeypatch, document)
    result = create_provider("paddleocr_api", {"model": "PaddleOCR-VL-1.6"}).recognize(page)
    assert {(c.eye, c.raw_value) for c in result.claims if c.field == "al"} == {("OD", "23.62"), ("OS", "23.41")}
    assert all(c.bbox is None for c in result.claims)
    ambiguous = {"prunedResult": {"parsing_res_list": [{"block_content": "ZEISS IOLMaster 700\nAL: 23.62 mm"}]}}
    output = paddle_api._document_claims(ambiguous, page)
    assert output.supported and not output.claims


def test_vl_declared_html_rowspan_and_explicit_axis_subheaders(page):
    content = "<table><tr><td rowspan='2'>Field</td><td colspan='2'>OD</td><td colspan='2'>OS</td></tr><tr><td>D</td><td>°</td><td>D</td><td>°</td></tr><tr><td>K1</td><td>44.91</td><td>165</td><td>45.21</td><td>71</td></tr></table>"
    tables = paddle_api._tables(content)
    claims = paddle_api._table_claims(tables[0], None)
    values = {(c.field, c.eye): c for c in claims}
    assert values["k1_power", "OD"].raw_value == "44.91"
    assert values["k1_axis", "OD"].raw_value == "165"
    assert values["k1_axis", "OS"].raw_value == "71"


def test_vl_unrelated_document_cannot_become_supported(page):
    output = paddle_api._document_claims({"prunedResult": {"parsing_res_list": [
        {"block_content": "Unrelated invoice\nOD AL: 23.62 mm"}]}}, page)
    assert output.supported is False and output.claims == []

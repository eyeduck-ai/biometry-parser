"""PaddleOCR official hosted jobs API, without a local Paddle dependency.

Protocol reference: PaddlePaddle/PaddleOCR paddleocr/_api_client (2026).
Only rendered page images are uploaded. Service result URLs are never stored or
given the access token; their JSONL is treated as untrusted recognition data.
OCR responses supply tokens, positions, and literal markers. Document-model
table claims may include blank candidates; the kit decides merged source states
and verification, rather than asking each model to implement that taxonomy.
"""
from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request
from uuid import uuid4

from ..models import Box, EYE_FIELDS, FieldClaim, Issue, PageInput, ProviderOutput, Span
from ..parsing import _DASH_MARKER, _DATE, _LABEL, _NUMBER, _UNIT, _label_field
from . import ProviderError
from .vision import urlopen
from ..pricing import price_for

DEFAULT_BASE_URL = "https://paddleocr.aistudio-app.com"
OCR_MODELS = {"PP-OCRv5", "PP-OCRv5-latin", "PP-OCRv6"}
DOCUMENT_MODELS = {"PP-StructureV3", "PaddleOCR-VL", "PaddleOCR-VL-1.5", "PaddleOCR-VL-1.6"}
_MAX_RESPONSE = 8 * 1024 * 1024
_RESULT_DOMAINS = ("bcebos.com", "baidubce.com", "aistudio-app.com")
_EYE = re.compile(r"^(?:(?:right|left)\s+)?(OD|OS)(?:\s+(?:right|left))?$", re.I)


def _invalid(message="PaddleOCR API response did not match the expected contract."):
    return ProviderError("invalid_provider_response", message)


def _endpoint(value: str) -> str:
    parts = urlsplit(value)
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment):
        raise ProviderError("invalid_configuration", "Configure a PaddleOCR service base URL without embedded credentials.")
    if parts.scheme != "https" and parts.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ProviderError("invalid_configuration", "Remote PaddleOCR services require HTTPS.")
    return value.rstrip("/")


def _result_url(value, allowed_hosts: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise _invalid()
    parts = urlsplit(value)
    host = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError:
        raise _invalid() from None
    if (parts.scheme != "https" or not host or parts.username or parts.password or parts.fragment
            or port not in {None, 443} or not any(host == h or host.endswith("." + h) for h in allowed_hosts)):
        raise _invalid("PaddleOCR API returned an untrusted result location.")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            raise _invalid("PaddleOCR API returned an untrusted result location.")
    else:
        raise _invalid("PaddleOCR API returned an untrusted result location.")
    return value


def _request_bytes(url, *, method="GET", body=None, headers=None, timeout=30, deadline=None):
    remaining = deadline - time.monotonic() if deadline is not None else timeout
    if remaining <= 0:
        raise ProviderError("provider_timeout", "PaddleOCR API exceeded its total time limit.")
    try:
        with urlopen(Request(url, data=body, headers=headers or {}, method=method), timeout=min(timeout, remaining)) as response:
            raw = response.read(_MAX_RESPONSE + 1)
        if len(raw) > _MAX_RESPONSE:
            raise _invalid("PaddleOCR API response exceeded the size limit.")
        if deadline is not None and time.monotonic() >= deadline:
            raise ProviderError("provider_timeout", "PaddleOCR API exceeded its total time limit.")
        return raw
    except HTTPError as exc:
        raise ProviderError("api_rate_limited" if exc.code == 429 else "api_http_error", f"PaddleOCR API returned HTTP {exc.code}.") from None
    except (URLError, TimeoutError, OSError):
        raise ProviderError("api_transport_error", "PaddleOCR API could not be reached within the configured limits.") from None


def _data(raw: bytes) -> dict:
    try:
        payload = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        if not isinstance(payload, dict) or payload.get("code", 0) not in {0, None}:
            raise ValueError()
        data = payload["data"]
        if not isinstance(data, dict):
            raise ValueError()
        return data
    except (ValueError, KeyError, TypeError, UnicodeError):
        # Service error messages and pre-signed URLs may contain private data.
        raise _invalid() from None


def _bbox(coords, width: int, height: int) -> Box:
    try:
        if len(coords) == 4 and all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in coords):
            points = [[coords[0], coords[1]], [coords[2], coords[3]]]
        else:
            points = coords
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError()
        xs, ys = [], []
        for p in points:
            if not isinstance(p, list) or len(p) != 2 or any(isinstance(v, bool) for v in p):
                raise ValueError()
            x, y = float(p[0]), float(p[1])
            if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= width or not 0 <= y <= height:
                raise ValueError()
            xs.append(x / width)
            ys.append(y / height)
        return Box(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))
    except (ValueError, TypeError, IndexError):
        raise _invalid("PaddleOCR API returned invalid source coordinates.") from None


def _ocr_spans(pruned: dict, page: PageInput) -> list[Span] | None:
    if "rec_texts" not in pruned:
        return None
    texts = pruned["rec_texts"]
    boxes = pruned.get("rec_polys", pruned.get("rec_boxes"))
    scores = pruned.get("rec_scores")
    if (not isinstance(texts, list) or not isinstance(boxes, list) or len(texts) != len(boxes)
            or scores is not None and (not isinstance(scores, list) or len(scores) != len(texts))):
        raise _invalid()
    spans = []
    for index, (text, box) in enumerate(zip(texts, boxes)):
        if not isinstance(text, str):
            raise _invalid()
        score = None if scores is None else scores[index]
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
            raise _invalid()
        spans.append(Span(text=text, bbox=_bbox(box, page.width, page.height), confidence=score,
                          confidence_scale="0..1 (uncalibrated)" if score is not None else None))
    return spans


class _Tables(HTMLParser):
    """Expand only HTML-declared row/column spans; no geometric cell invention."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.rows, self.cells, self.cell, self.depth = [], [], [], None, 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                self.rows = []
        elif self.depth == 1 and tag == "tr":
            self.cells = []
        elif self.depth == 1 and tag in {"td", "th"}:
            attributes = dict(attrs)
            try:
                rowspan, colspan = int(attributes.get("rowspan", 1)), int(attributes.get("colspan", 1))
                if not 1 <= rowspan <= 32 or not 1 <= colspan <= 32:
                    raise ValueError()
            except (ValueError, TypeError):
                raise _invalid("PaddleOCR API returned an unsupported table structure.") from None
            self.cell = {"text": "", "rowspan": rowspan, "colspan": colspan}
        elif self.cell is not None and tag == "br":
            self.cell["text"] += "\n"

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"] += data

    def handle_endtag(self, tag):
        if self.depth == 1 and tag in {"td", "th"} and self.cell is not None:
            self.cells.append(self.cell)
            self.cell = None
        elif self.depth == 1 and tag == "tr":
            self.rows.append(self.cells)
        elif tag == "table":
            if self.depth == 1:
                grid = {}
                for row_index, row in enumerate(self.rows):
                    column = 0
                    for cell in row:
                        while (row_index, column) in grid:
                            column += 1
                        for y in range(row_index, row_index + cell["rowspan"]):
                            for x in range(column, column + cell["colspan"]):
                                if y > 500 or x > 64 or (y, x) in grid:
                                    raise _invalid("PaddleOCR API table exceeded the structural limits.")
                                grid[y, x] = cell["text"].strip()
                        column += cell["colspan"]
                if grid:
                    max_y, max_x = max(y for y, _ in grid), max(x for _, x in grid)
                    self.tables.append([[grid.get((y, x), "") for x in range(max_x + 1)] for y in range(max_y + 1)])
            self.depth = max(0, self.depth - 1)


def _tables(text: str) -> list[list[list[str]]]:
    parser = _Tables()
    parser.feed(text)
    tables = parser.tables
    pending = []
    for line in text.splitlines() + [""]:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                pending.append(cells)
        elif pending:
            tables.append(pending)
            pending = []
    return tables


def _unit(text: str) -> str | None:
    match = re.search(r"(?<![A-Za-z])(?:mm|[µμu]m|D|degrees?|deg|°)(?![A-Za-z])", text, re.I)
    return match.group() if match else None


def _field(text: str) -> str | None:
    direct = _label_field(re.sub(r"\([^)]*\)|\[[^]]*\]", "", text).strip())
    if direct:
        return direct
    match = _LABEL.match(text.strip())
    return _label_field(match.group("label")) if match else None


def _value_claims(field, label, value, eye, box, unit_hint=None, *, empty_cell=False):
    raw = unescape(value).strip()
    # A cell may repeat its label; strip that explicit label before reading it.
    label_match = _LABEL.match(raw)
    if label_match and _label_field(label_match.group("label")) == field:
        raw = raw[label_match.end():].strip()
    raw_text = f"{label}: {value}"
    common = {"field": field, "eye": eye if field in EYE_FIELDS else None, "raw_label": label,
              "raw_text": raw_text, "bbox": box, "flags": ["service_table_or_block_evidence"],
              "state_evidence": "provider_assertion"}
    # An explicitly empty table cell is a model assertion of whitespace, not
    # confirmed absence. A naked label/empty OCR line remains unreadable.
    empty_state = "blank" if not raw and empty_cell else "unreadable"
    if field == "patient_id":
        match = re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/\-]*", raw)
        return [FieldClaim(**common, raw_value=match.group() if match else None, missing=None if match else empty_state)]
    if field == "gender":
        match = re.fullmatch(r"Male|Female|M|F|Other|Unknown", raw, re.I)
        return [FieldClaim(**common, raw_value=match.group() if match else None, missing=None if match else empty_state)]
    if field == "measurement_date":
        match = _DATE.fullmatch(raw)
        return [FieldClaim(**common, raw_value=match.group() if match else None, missing=None if match else empty_state)]
    match = _NUMBER.match(raw)
    missing, number, unit = None, None, None
    if match:
        number = match.group("value")
        if _DASH_MARKER.fullmatch(number):
            missing, number = "not_measured", None
            common["state_evidence"] = "explicit_marker"
        else:
            unit_match = _UNIT.match(raw[match.end():])
            unit = unit_match.group(1) if unit_match else unit_hint or _unit(label)
    else:
        missing = empty_state
    whole_row_blank = missing == "blank" and field.endswith("_power") and bool(re.search(r"@|\baxis\b|°", label, re.I))
    if whole_row_blank:
        common["flags"].append("whole_keratometry_row_blank")
    claims = [FieldClaim(**common, raw_value=number, unit=unit, missing=missing)]
    if field in {"k1_power", "k2_power", "tk1_power", "tk2_power", "corneal_astigmatism"}:
        axis = re.search(r"@\s*([+\-−]?\d+(?:[.,]\d+)?)\s*°?", raw)
        axis_field = field.removesuffix("_power") + "_axis" if field.endswith("_power") else "corneal_astigmatism_axis"
        # An explicit dash row denotes the paired missing power and axis.
        if axis or missing == "not_measured" or whole_row_blank:
            claims.append(FieldClaim(**{**common, "field": axis_field}, raw_value=axis.group(1) if axis else None,
                                     unit="degree", missing=missing))
    return claims


def _table_claims(rows, box):
    claims, eyes, unit_hints, column_fields = [], {}, {}, {}
    for row in rows:
        for i, cell in enumerate(row):
            match = _LABEL.match(cell.strip())
            field = _label_field(match.group("label")) if match else None
            if field in {"patient_id", "gender", "measurement_date"}:
                tail = cell.strip()[match.end():].strip()
                value = tail if tail else row[i + 1] if i + 1 < len(row) else ""
                claims.extend(_value_claims(field, match.group("label"), value, None, box, empty_cell=not tail and i + 1 < len(row)))
        eye_headers = {i: _EYE.fullmatch(c.strip()).group(1).upper() for i, c in enumerate(row) if _EYE.fullmatch(c.strip())}
        if len(eye_headers) >= 2 and len(set(eye_headers.values())) == 2:
            eyes = eye_headers
            continue
        if eyes:
            label = next((c for i, c in enumerate(row) if i not in eyes and _field(c)), None)
            field = _field(label) if label else None
            if field:
                for i, eye in eyes.items():
                    if i >= len(row):
                        continue
                    effective = field
                    if unit_hints.get(i) in {"°", "degree", "degrees", "deg"} and field.endswith("_power"):
                        effective = field.removesuffix("_power") + "_axis"
                    claims.extend(_value_claims(effective, label, row[i], eye, box, unit_hints.get(i), empty_cell=True))
                continue
            for i, cell in enumerate(row):
                if i in eyes and _unit(cell):
                    unit_hints[i] = _unit(cell)
        # Also support tables where OD and OS are rows, with labeled columns.
        first_eye = _EYE.fullmatch(row[0].strip()) if row else None
        if first_eye and column_fields:
            for i, (field, label) in column_fields.items():
                if i < len(row):
                    claims.extend(_value_claims(field, label, row[i], first_eye.group(1).upper(), box, empty_cell=True))
        elif not eyes:
            last_power = None
            fields = {}
            for i, cell in enumerate(row):
                field = _field(cell)
                if field:
                    fields[i] = (field, cell)
                    last_power = field if field.endswith("_power") else None
                elif last_power and re.search(r"\baxis\b|°", cell, re.I):
                    fields[i] = (last_power.removesuffix("_power") + "_axis", cell)
            if fields:
                column_fields = fields
    return claims


def _document_claims(item: dict, page: PageInput) -> ProviderOutput:
    pruned = item.get("prunedResult")
    if not isinstance(pruned, dict):
        raise _invalid()
    blocks = pruned.get("parsing_res_list", [])
    if not isinstance(blocks, list):
        raise _invalid()
    markdown = item.get("markdown", {})
    markdown_text = markdown.get("text", "") if isinstance(markdown, dict) else ""
    if not isinstance(markdown_text, str):
        raise _invalid()
    source_blocks = []
    for block in blocks:
        if not isinstance(block, dict) or not isinstance(block.get("block_content"), str):
            raise _invalid()
        box = _bbox(block["block_bbox"], page.width, page.height) if block.get("block_bbox") is not None else None
        source_blocks.append((block["block_content"], box))
    text = "\n".join(content for content, _ in source_blocks) or markdown_text
    compact = re.sub(r"\s+", "", re.sub(r"<[^>]*>", " ", text)).lower()
    supported = "iolmaster700" in compact or "zeiss" in compact and "biometricvalues" in compact
    claims = []
    if supported:
        tables = [(table, box) for content, box in source_blocks for table in _tables(content)]
        if not tables:
            tables = [(table, None) for table in _tables(markdown_text)]
        for table, box in tables:
            claims.extend(_table_claims(table, box))
        # Metadata and inline explicitly named eyes do not need a table cell box.
        for content, box in source_blocks or [(markdown_text, None)]:
            plain = unescape(re.sub(r"<[^>]*>", "\n", content))
            for line in plain.splitlines():
                matches = list(_LABEL.finditer(line))
                explicit_eyes = {m.group(1).upper() for m in re.finditer(r"\b(OD|OS)\b", line, re.I)}
                for index, match in enumerate(matches):
                    field = _label_field(match.group("label"))
                    end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
                    value = line[match.end():end].strip()
                    if field not in EYE_FIELDS or len(explicit_eyes) == 1:
                        claims.extend(_value_claims(field, match.group("label"), value,
                                                    next(iter(explicit_eyes)) if explicit_eyes else None, box))
    unique = {}
    for claim in claims:
        unique.setdefault((claim.field, claim.eye, claim.raw_value, claim.missing, claim.raw_text), claim)
    issues = [Issue(code="paddle_document_layout_requires_review", message="Document API claims use explicit table/eye labels; unsupported layouts and cell-level geometry require review.", page_number=page.page_number)]
    return ProviderOutput(kind="claims", supported=supported, claims=list(unique.values()), issues=issues,
                          metadata={"raw_document_text": text, "evidence_precision": "block_or_page", "report_type": "structured"})


class PaddleOCRAPIProvider:
    def __init__(self, config: dict):
        self.config = config
        self.options = config.get("options", {})
        self.model = config.get("model") or "PP-OCRv6"
        if self.model not in OCR_MODELS | DOCUMENT_MODELS:
            raise ProviderError("invalid_configuration", "Configure a supported official PaddleOCR API model ID.")
        self.base_url = _endpoint(config.get("base_url") or DEFAULT_BASE_URL)
        self.key = os.environ.get(config.get("api_key_env") or "PADDLEOCR_ACCESS_TOKEN", "")
        if not self.key or any(ord(c) < 32 or ord(c) == 127 for c in self.key):
            raise ProviderError("credentials_unavailable", "PaddleOCR AI Studio access token is not configured.")
        self.timeout = float(config.get("timeout_seconds", 120))
        self.request_timeout = float(self.options.get("request_timeout_seconds", min(30, self.timeout)))
        self.poll_interval = float(self.options.get("poll_interval_seconds", 1))
        hosts = self.options.get("result_host_allowlist", [])
        if (not 0 < self.timeout <= 3600 or not 0 < self.request_timeout <= self.timeout or not .05 <= self.poll_interval <= 10
                or not isinstance(hosts, list) or any(not isinstance(h, str) or not re.fullmatch(r"[A-Za-z0-9.-]+", h) for h in hosts)):
            raise ProviderError("invalid_configuration", "Invalid PaddleOCR API request limits or result hosts.")
        self.allowed_result_hosts = (*_RESULT_DOMAINS, urlsplit(self.base_url).hostname.lower(), *(h.lower().strip(".") for h in hosts))

    def describe(self):
        return {"id": "paddleocr_api", "family": "paddle_ppocr" if self.model in OCR_MODELS or self.model == "PP-StructureV3" else "paddle_ocr_vl",
                "kind": "spans" if self.model in OCR_MODELS else "claims", "remote": True, "model_id": self.model,
                "protocol": "paddleocr-official-jobs-v2", "resource_scope": "local_client_only", "server_resources_available": False}

    def recognize(self, page: PageInput) -> ProviderOutput:
        started = time.monotonic()
        prepared = time.perf_counter()
        network = {'transport_seconds':0.,'request_bytes':0,'response_bytes':0,'http_requests':0}
        def request(url,**kwargs):
            tick=time.perf_counter()
            raw=_request_bytes(url,**kwargs)
            network['transport_seconds']+=time.perf_counter()-tick
            network['request_bytes']+=len(kwargs.get('body') or b'')
            network['response_bytes']+=len(raw)
            network['http_requests']+=1
            return raw
        deadline = started + self.timeout * .97
        path = Path(page.image_path)
        if path.stat().st_size > 50 * 1024 * 1024:
            raise ProviderError("invalid_configuration", "Rendered PaddleOCR API page exceeds the upload size limit.")
        optional = {"useDocOrientationClassify": False, "useDocUnwarping": False, "visualize": False}
        if self.model in OCR_MODELS:
            optional.update(useTextlineOrientation=False, textRecScoreThresh=0.0)
        elif self.model == "PP-StructureV3":
            optional.update(useTableRecognition=True, useTextlineOrientation=False, textRecScoreThresh=0.0,
                            useFormulaRecognition=False, useChartRecognition=False, prettifyMarkdown=False)
        else:
            optional.update(useLayoutDetection=True, prettifyMarkdown=False)
        boundary = "biometry-" + uuid4().hex
        pieces = []
        for name, value in (("model", self.model), ("optionalPayload", json.dumps(optional))):
            pieces.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        pieces.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="page.png"\r\nContent-Type: image/png\r\n\r\n'.encode())
        pieces.extend((path.read_bytes(), f"\r\n--{boundary}--\r\n".encode()))
        jobs = self.base_url + "/api/v2/ocr/jobs"
        headers = {"Authorization": "Bearer " + self.key}
        preprocessing_seconds=time.perf_counter()-prepared
        data = _data(request(jobs, method="POST", body=b"".join(pieces),
                                   headers={**headers, "Content-Type": "multipart/form-data; boundary=" + boundary},
                                   timeout=self.request_timeout, deadline=deadline))
        job_id = data.get("jobId")
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", job_id):
            raise _invalid()
        polls = 0
        while True:
            data = _data(request(jobs + "/" + job_id, headers=headers, timeout=self.request_timeout, deadline=deadline))
            polls += 1
            state = data.get("state")
            if state == "done":
                location = data.get("resultUrl")
                location = location.get("jsonUrl") if isinstance(location, dict) else None
                result_url = _result_url(location, self.allowed_result_hosts)
                if self.key in result_url:
                    raise _invalid("PaddleOCR API returned an invalid result location.")
                raw = request(result_url, headers={}, timeout=self.request_timeout, deadline=deadline)
                break
            if state == "failed":
                raise ProviderError("api_job_failed", "PaddleOCR API recognition job failed.")
            if state not in {"pending", "running"}:
                raise _invalid()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderError("provider_timeout", "PaddleOCR API exceeded its total time limit.")
            time.sleep(min(self.poll_interval, remaining))
        try:
            documents = [json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError())) for line in raw.decode("utf-8").splitlines() if line.strip()]
            if not documents:
                raise ValueError()
            key = "ocrResults" if self.model in OCR_MODELS else "layoutParsingResults"
            pages = [item for document in documents for item in document["result"][key]]
            if len(pages) != 1 or not isinstance(pages[0], dict):
                raise ValueError()
            item = pages[0]
            pruned = item["prunedResult"]
            if not isinstance(pruned, dict):
                raise ValueError()
        except (KeyError, ValueError, TypeError, UnicodeError):
            raise _invalid() from None
        spans = _ocr_spans(pruned, page)
        if spans is None and isinstance(pruned.get("overall_ocr_res"), dict):
            spans = _ocr_spans(pruned["overall_ocr_res"], page)
        if self.model in OCR_MODELS and spans is None:
            raise _invalid()
        output = ProviderOutput(kind="spans", spans=spans) if spans is not None else _document_claims(item, page)
        output.metadata.update(self.describe())
        output.metadata.update(api_poll_count=polls, api_elapsed_seconds=time.monotonic() - started,
                               api_usage_available=False, usage_complete=False,
                               **network,preprocessing_seconds=preprocessing_seconds,pricing=price_for('paddleocr_api',self.model),
                               billing_tier='hosted_free_quota',actual_charge_usd=None,
                               transport_scope='http_roundtrip_including_server_queue_and_inference')
        # Actual response type may differ from the model's ordinary document mode.
        output.metadata["kind"] = output.kind
        return output

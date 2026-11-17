"""Conservative, geometry-aware IOLMaster 700 report interpretation.

OCR spans and VLM claims use the same normalization rules. This module never
computes a missing measurement or resolves disagreements between providers.
Backends provide page evidence; they need not implement every final data state.
Provider blank claims remain candidates, provider absence claims are downgraded,
and evidence categories are assigned here rather than trusted from model output.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import re
from statistics import median

from .models import (Box, EYE_FIELDS, FIELDS, UNITS, FieldClaim, Issue,
                     Observation, PageInput, ParsedPage, ProviderOutput, Span)

_LABEL = re.compile(
    r"(?<![A-Za-z])(?P<label>Patient\s*ID|Date\s+of\s+measurement|Gender|"
    r"Avg\.?\s*K|Ast\.?\s*K|[Δ∆]\s*K|TK[12]|K[12]|ACD|CCT|WTW|AL|LT|SE|Pupil|P)(?![A-Za-z0-9])\s*:?", re.I)
_DASH_MARKER = re.compile(r"(?:[-‐‑‒–—−]\s*){2,}")
_NUMBER = re.compile(r"^\s*[:=]?\s*(?P<prefix>[!*]?\s*)(?P<value>(?:[-‐‑‒–—−]\s*){2,}|[+\-−]?\d+(?:[.,]\d+)?)(?P<suffix>\s*[!*]?)")
_UNIT = re.compile(r"^\s*(mm|[µμu]m|D|degrees?|deg|°)(?![A-Za-z])", re.I)
_DATE = re.compile(r"^(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4})(?!\d)")
_ALIASES = {"al": "al", "acd": "acd", "lt": "lt", "cct": "cct", "wtw": "wtw", "se": "mean_k", "avgk": "mean_k", "astk": "corneal_astigmatism", "δk": "corneal_astigmatism", "∆k": "corneal_astigmatism", "p": "pupil", "pupil": "pupil", "patientid": "patient_id", "gender": "gender", "dateofmeasurement": "measurement_date"}


def _label_field(label: str) -> str | None:
    compact = re.sub(r"[\s.:]+", "", label).lower()
    if compact in FIELDS:
        return compact
    if compact in {"k1", "k2", "tk1", "tk2"}:
        return compact + "_power"
    return _ALIASES.get(compact)


def _union(spans: list[Span]) -> Box:
    return Box(x0=min(s.bbox.x0 for s in spans), y0=min(s.bbox.y0 for s in spans),
               x1=max(s.bbox.x1 for s in spans), y1=max(s.bbox.y1 for s in spans))


@dataclass
class _Row:
    spans: list[Span]

    @property
    def y(self):
        return median((s.bbox.y0 + s.bbox.y1) / 2 for s in self.spans)

    def string(self):
        spans = sorted(self.spans, key=lambda s: s.bbox.x0)
        pieces, intervals, offset = [], [], 0
        for span in spans:
            if pieces:
                offset += 1
            pieces.append(span.text)
            intervals.append((offset, offset + len(span.text), span))
            offset += len(span.text)
        return " ".join(pieces), intervals


def _rows(spans: list[Span]) -> list[_Row]:
    # Very tall side-watermarks are not horizontal report content.
    spans = [s for s in spans if s.text.strip() and s.bbox.y1 - s.bbox.y0 < .075]
    rows: list[_Row] = []
    for span in sorted(spans, key=lambda s: ((s.bbox.y0 + s.bbox.y1) / 2, s.bbox.x0)):
        center = (span.bbox.y0 + span.bbox.y1) / 2
        height = max(.001, span.bbox.y1 - span.bbox.y0)
        row = next((r for r in reversed(rows[-4:]) if abs(center - r.y) <= max(.002, min(height, median(x.bbox.y1-x.bbox.y0 for x in r.spans)) * .65)), None)
        if row is None:
            rows.append(_Row([span]))
        else:
            row.spans.append(span)
    return sorted(rows, key=lambda r: r.y)


def _parse_date(raw: str, order: str) -> str:
    if order == "reject_ambiguous":
        order = "auto"
    parts = re.split(r"[/.\-]", raw)
    if len(parts[0]) == 4:
        year, month, day = map(int, parts)
    else:
        first, second, year = map(int, parts)
        if order == "mdy":
            month, day = first, second
        elif order == "dmy":
            day, month = first, second
        elif order == "auto":
            if first <= 12 and second <= 12 and first != second:
                raise ValueError("ambiguous date requires date_order")
            day, month = (first, second) if first > 12 else (second, first)
        else:
            raise ValueError("date_order must be mdy, dmy or reject_ambiguous")
    return date(year, month, day).isoformat()


def _normalize(claim: FieldClaim, page: PageInput, *, profile_id: str,
               model_id: str, family: str, date_order: str, provider_id: str,
               confidence=None, confidence_scale=None) -> tuple[Observation | None, list[Issue]]:
    field = _label_field(claim.field) or (claim.field if claim.field in FIELDS else None)
    if field is None:
        return None, [Issue(code="unknown_field", message="Unrecognized field claim", page_number=page.page_number, profile_id=profile_id)]
    observation = Observation(field=field, eye=claim.eye, value=None, unit=UNITS.get(field),
                              status=claim.missing or "present", page_number=page.page_number,
                              bbox=claim.bbox, raw_label=claim.raw_label, raw_text=claim.raw_text,
                              provider_id=provider_id, profile_id=profile_id, model_id=model_id,
                              family=family, role=claim.role, flags=claim.flags,
                              confidence=confidence, confidence_scale=confidence_scale,
                              state_evidence=claim.state_evidence)
    issues = []

    def invalid(code, message):
        observation.status = "unreadable"
        observation.value = None
        observation.state_evidence = "label_only" if claim.raw_label else "provider_assertion"
        issues.append(Issue(code=code, message=message, page_number=page.page_number, field=field, eye=claim.eye, profile_id=profile_id))

    if field in EYE_FIELDS and claim.eye is None:
        invalid("eye_unresolved", "No reliable eye header for measurement")
    if field not in EYE_FIELDS and claim.eye is not None:
        invalid("identity_eye_invalid", "Patient metadata must not be eye-specific")
    if issues:
        return observation, issues
    if claim.missing:
        if claim.missing == "absent":
            # A provider failing to find a field cannot confirm its absence.
            # Human truth and trusted template checks use separate call paths.
            observation.status = "not_found"
            observation.state_evidence = "provider_assertion"
            observation.flags.append("absence_unconfirmed")
            issues.append(Issue(code="absence_unconfirmed", message="Provider absence assertion requires human or verified-template confirmation.",
                                page_number=page.page_number, field=field, eye=claim.eye, profile_id=profile_id))
        elif claim.missing == "blank":
            if not claim.raw_label.strip() or claim.bbox is None or claim.bbox.x1 <= claim.bbox.x0 or claim.bbox.y1 <= claim.bbox.y0:
                invalid("blank_unconfirmed", "A blank-field claim requires a visible label and source region.")
                observation.flags.append("blank_unconfirmed")
            else:
                observation.state_evidence = "provider_assertion"
                observation.flags.append("blank_requires_review")
        elif claim.missing in {"not_found", "not_reported"}:
            observation.status = "not_found"
            observation.state_evidence = "none"
            if claim.missing == "not_reported":
                observation.flags.append("legacy_missing_semantics")
        elif claim.missing == "not_measured":
            observation.state_evidence = "explicit_marker" if _DASH_MARKER.search(claim.raw_text) else "provider_assertion"
        elif claim.missing == "unreadable":
            observation.state_evidence = "label_only" if claim.raw_label else "provider_assertion"
        return observation, issues
    raw = (claim.raw_value or "").strip()
    if not raw:
        observation.status = "unreadable" if claim.raw_label.strip() else "not_found"
        observation.state_evidence = "label_only" if claim.raw_label.strip() else "none"
        return observation, issues
    if field in EYE_FIELDS and claim.eye is not None and _DASH_MARKER.fullmatch(raw):
        # Structured APIs sometimes return the printed marker as raw_value
        # rather than filling the explicit missing-status property.
        observation.status = "not_measured"
        observation.state_evidence = "explicit_marker"
        return observation, issues
    observation.state_evidence = "provider_assertion"
    try:
        if field == "patient_id":
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/\-]*", raw):
                raise ValueError("invalid patient ID")
            observation.value = raw
        elif field == "gender":
            normalized = {"male": "male", "m": "male", "female": "female", "f": "female", "other": "other", "unknown": "unknown"}.get(raw.lower())
            if normalized is None:
                raise ValueError("unsupported gender text")
            observation.value = normalized
        elif field == "measurement_date":
            if not _DATE.fullmatch(raw):
                raise ValueError("invalid measurement date")
            observation.value = _parse_date(raw, date_order)
        else:
            # Without an explicit numeric locale, a comma could be a grouping
            # or decimal separator. Preserve it as uncertain rather than guess.
            number = raw.replace("−", "-")
            if not re.fullmatch(r"[+\-]?\d+(?:\.\d+)?", number):
                raise ValueError("invalid number")
            value = Decimal(number)
            unit = (claim.unit or "").strip().replace("µ", "u").replace("μ", "u").lower()
            expected = UNITS[field]
            if expected == "degree":
                if unit not in {"degree", "degrees", "deg", "°"}:
                    raise ValueError("missing or unsupported axis unit")
                if not Decimal(0) <= value <= Decimal(180):
                    raise ValueError("axis must be within 0..180 degrees")
            elif expected == "mm":
                if unit == "um":
                    value /= 1000
                elif unit != "mm":
                    raise ValueError("missing or unsupported length unit")
            elif expected == "um":
                if unit == "mm":
                    value *= 1000
                elif unit != "um":
                    raise ValueError("missing or unsupported corneal thickness unit")
            elif unit != "d":
                raise ValueError("missing or unsupported keratometry unit")
            observation.value = value
    except (ValueError, InvalidOperation) as error:
        invalid("normalization_failed", str(error))
    if any(i.code in {"eye_unresolved", "identity_eye_invalid"} for i in issues):
        observation.value = None
        observation.status = "unreadable"
    return observation, issues


def _spans_claims(rows: list[_Row], report_type: str) -> tuple[list[tuple[FieldClaim, float | None, str | None]], list[Issue]]:
    claims, issues = [], []
    eye_markers = []
    body_y = 1.0
    for row in rows:
        text, _ = row.string()
        if any(_label_field(m.group("label")) in EYE_FIELDS for m in _LABEL.finditer(text)) and _NUMBER.search(text[_LABEL.search(text).end():] if _LABEL.search(text) else ""):
            body_y = min(body_y, row.y)
        for span in row.spans:
            match = re.fullmatch(r"\s*(?:(right|left)\s+)?(OD|OS)(?:\s+(right|left))?\s*", span.text, re.I)
            if match:
                eye_markers.append((row.y, (span.bbox.x0 + span.bbox.x1)/2, match.group(2).upper()))
    candidates = [m for m in eye_markers if m[0] <= body_y + .025]
    if not candidates:
        candidates = eye_markers
    if candidates:
        # The last heading before values wins over earlier warning banners.
        heading_y = max(m[0] for m in candidates)
        candidates = [m for m in candidates if abs(m[0]-heading_y) < .025]
    by_eye = {eye: x for _, x, eye in candidates}
    if len(by_eye) == 1:
        # A dropped OS/OD heading on a bilateral OCR page must not turn the
        # entire other column into a monocular result. Repeated labeled
        # measurements on the same physical row establish that ambiguity.
        for row in rows:
            text, intervals = row.string()
            fields_on_row: dict[str, list[float]] = {}
            for match in _LABEL.finditer(text):
                field = _label_field(match.group("label"))
                if field not in EYE_FIELDS:
                    continue
                label_spans = [s for a,b,s in intervals if b > match.start() and a < match.end()]
                if label_spans:
                    fields_on_row.setdefault(field, []).append(label_spans[0].bbox.x0)
            if any(len(xs) > 1 and max(xs)-min(xs) > .25 for xs in fields_on_row.values()):
                by_eye = {}
                break
    seen: set[tuple[str, str | None]] = set()
    for row in rows:
        text, intervals = row.string()
        if re.search(r"\bIOL\b", text, re.I) and re.search(r"\bRef\b|\bRes\.?\s*SE\b", text, re.I):
            # Even a table whose numerical rows were unreadable must not turn
            # its IOL SE / Res.SE headings into measured corneal SE candidates.
            continue
        matches = list(_LABEL.finditer(text))
        # Axial diagrams print a horizontal AL/CCT/ACD/LT legend. Those
        # label-only table/diagram headers are not unreadable summary cells.
        legend_fields = {_label_field(item.group("label")) for item in matches}
        label_legend = len(legend_fields & set(EYE_FIELDS)) >= 3 and not any(
            _NUMBER.match(text[item.end():(matches[i+1].start() if i+1 < len(matches) else len(text))].strip())
            for i, item in enumerate(matches))
        for index, match in enumerate(matches):
            field = _label_field(match.group("label"))
            if field is None:
                continue
            end = matches[index+1].start() if index+1 < len(matches) else len(text)
            tail = text[match.end():end].strip()
            label_spans = [s for a,b,s in intervals if b > match.start() and a < match.end()]
            if not label_spans:
                continue
            x = (label_spans[0].bbox.x0 + label_spans[0].bbox.x1)/2
            eye = None
            if field in EYE_FIELDS:
                if len(by_eye) == 1:
                    eye = next(iter(by_eye))
                elif len(by_eye) == 2:
                    left_eye, right_eye = sorted(by_eye, key=by_eye.get)
                    middle = (by_eye[left_eye] + by_eye[right_eye])/2
                    eye = left_eye if x < middle else right_eye
                # Do not interpret field-like words above the eye heading.
                if candidates and row.y < min(m[0] for m in candidates) - .01:
                    continue
            raw_value, unit, missing = None, None, None
            if field == "patient_id":
                value = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._/\-]*)", tail)
                raw_value = value.group(1) if value else None
            elif field == "gender":
                value = re.match(r"^\s*(Male|Female|Unknown|Other|M|F)\b", tail, re.I)
                raw_value = value.group(1) if value else None
            elif field == "measurement_date":
                value = _DATE.match(tail)
                raw_value = value.group(1) if value else None
            else:
                value = _NUMBER.match(tail)
                if not value:
                    if label_legend or match.group("label") == "p" or field == "pupil" and re.match(r"values\b", tail, re.I):
                        # A lower-case footnote/IOL-table symbol is not a
                        # recognizable P/Pupil measurement label by itself.
                        continue
                    # A visible label without readable value is distinct from
                    # a field that was not found. It does not prove whitespace
                    # in the original source (OCR may have dropped the value).
                    missing = "unreadable"
                else:
                    raw_value = value.group("value")
                    if _DASH_MARKER.fullmatch(raw_value):
                        raw_value, missing = None, "not_measured"
                    else:
                        unit_match = _UNIT.match(tail[value.end():])
                        unit = unit_match.group(1) if unit_match else None
            evidence_spans = [s for a,b,s in intervals if b > match.start() and a < end]
            bbox = _union(evidence_spans)
            flags = []
            if "*" in tail or "*" in text[max(0, match.start()-2):match.start()]:
                flags.append("manually_edited")
            if "!" in tail or "!" in text[max(0, match.start()-2):match.start()]:
                flags.append("borderline")
            role = "summary"
            # Biometric Values printouts repeat three individual SE/DeltaK
            # pairs beneath the labeled summary K1/K2 block.
            if report_type == "biometric" and field in {"mean_k", "corneal_astigmatism"} and (field, eye) in seen and re.search(r"\bSE\s*:?\s*[+\-]?\d", text, re.I) and re.search(r"[Δ∆]\s*K", text):
                role = "individual"
            claim = FieldClaim(field=field, eye=eye, raw_value=raw_value, unit=unit,
                               raw_label=match.group("label"), raw_text=text[match.start():end].strip(),
                               bbox=bbox, missing=missing, flags=flags, role=role,
                               state_evidence="explicit_marker" if missing == "not_measured" else
                                              "label_only" if raw_value is None else "provider_assertion")
            confidences = [s.confidence for s in evidence_spans if s.confidence is not None]
            confidence = min(confidences) if confidences else None
            scale = next((s.confidence_scale for s in evidence_spans if s.confidence_scale), None)
            claims.append((claim, confidence, scale))
            seen.add((field, eye))
            if field in {"k1_power", "k2_power", "tk1_power", "tk2_power", "corneal_astigmatism"}:
                axis_field = field.removesuffix("_power") + "_axis" if field.endswith("_power") else "corneal_astigmatism_axis"
                axis = re.search(r"@\s*([+\-−]?\d+(?:[.,]\d+)?)\s*°?", tail)
                axis_claim = claim.model_copy(update={"field": axis_field, "raw_value": axis.group(1) if axis else None,
                                                       "unit": "degree", "missing": missing or (None if axis else "not_found")})
                claims.append((axis_claim, confidence, scale))
    return claims, issues


def _recover_tk_markers(page: PageInput, claims):
    from .dash_recovery import METHOD, find_dash_triplet, open_source_image

    unresolved = [claim for claim, _, _ in claims if claim.field in {"tk1_power", "tk2_power"}
                  and claim.missing == "unreadable" and claim.bbox is not None and claim.eye is not None]
    if not unresolved:
        return claims
    image = open_source_image(page.image_path)
    if image is None:
        return claims
    corrections = {}
    try:
        for claim in unresolved:
            evidence = find_dash_triplet(image, claim.bbox)
            if evidence is not None:
                key = (claim.eye, claim.field.removesuffix("_power"), claim.bbox.model_dump_json())
                corrections[key] = evidence
    finally:
        image.close()
    recovered = []
    for claim, confidence, scale in claims:
        group = claim.field.removesuffix("_power").removesuffix("_axis")
        key = (claim.eye, group, claim.bbox.model_dump_json() if claim.bbox else "")
        evidence = corrections.get(key)
        if evidence is not None and claim.missing in {"unreadable", "not_reported"}:
            bbox = Box(x0=min(claim.bbox.x0, evidence.bbox.x0), y0=min(claim.bbox.y0, evidence.bbox.y0),
                       x1=max(claim.bbox.x1, evidence.bbox.x1), y1=max(claim.bbox.y1, evidence.bbox.y1))
            claim = claim.model_copy(update={"raw_value": None, "missing": "not_measured", "bbox": bbox,
                "raw_text": claim.raw_text + " | image marker: ---", "flags": [*claim.flags, METHOD],
                "state_evidence": "explicit_marker"})
            # The OCR engine's confidence is not a confidence for pixel recovery.
            confidence = scale = None
        recovered.append((claim, confidence, scale))
    return recovered


def recover_tk_observations(page: PageInput, observations: list[Observation]) -> list[Observation]:
    """Reprocess saved OCR evidence without rerunning or relabeling its model.

    Callers must record that recognition was reused and keep the original
    result. Only an unresolved TK label with an image-confirmed dash triplet can
    change. Positive measurements and other fields are copied unchanged.
    """
    claims = [(FieldClaim(field=o.field, eye=o.eye, raw_value=str(o.value) if o.value is not None else None,
                          unit=o.unit, raw_label=o.raw_label, raw_text=o.raw_text, bbox=o.bbox,
                          role=o.role, missing=o.status if o.status != "present" else None,
                          flags=list(o.flags), state_evidence=o.state_evidence), o.confidence, o.confidence_scale) for o in observations]
    recovered = _recover_tk_markers(page, claims)
    return [original.model_copy(update={"status": claim.missing or original.status,
             "value": None if claim.missing else original.value, "bbox": claim.bbox,
             "raw_text": claim.raw_text, "flags": claim.flags,
             "confidence": confidence, "confidence_scale": scale, "state_evidence": claim.state_evidence}, deep=True)
            for original, (claim, confidence, scale) in zip(observations, recovered)]


def _paired_tk_item_states(page: PageInput, output: ProviderOutput) -> list[FieldClaim]:
    """Preserve the whole TK item's explicit missing state for its paired axis.

    Some image-to-claims providers return ``TK1: ---`` as the power claim and
    omit its axis. On this scoped IOLMaster report contract the marker describes
    the whole TK item, rather than a measured power with an unknown numeric
    axis. This adds source-state evidence only; it never invents an axis value
    or an axis glyph location, and never replaces a model's substantive claim.
    """
    claims = list(output.claims)
    if output.kind != "claims" or output.supported is not True:
        return claims
    from .report_templates import inspect_report
    inspection = inspect_report([*page.native_spans, *output.spans])
    if inspection.get("explicit_unsupported_device"):
        return claims
    known_contract = output.metadata.get("report_contract") == "zeiss_iolmaster_700_image_claims_v2"
    known_report = inspection.get("supported") and inspection.get("template_id") == "zeiss_iolmaster_700"
    if not (known_contract or known_report):
        return claims

    def no_axis_evidence(claim):
        return (claim.missing in {"not_found", "not_reported"} and claim.raw_value is None
                and not claim.raw_label.strip() and not claim.raw_text.strip() and claim.bbox is None)

    def whole_item_marker(claim, item):
        if claim.missing != "not_measured" or claim.raw_value is not None:
            return False
        if not re.fullmatch(rf"{item}\s*:?", claim.raw_label.strip(), re.I):
            return False
        marker = re.sub(rf"^{item}\s*:?\s*", "", claim.raw_text.strip(), flags=re.I)
        return _DASH_MARKER.fullmatch(marker) is not None

    additions = []
    for claim in claims:
        if claim.field not in {"tk1_power", "tk2_power"} or claim.eye not in {"OD", "OS"} or claim.role != "summary":
            continue
        item = claim.field.removesuffix("_power").upper()
        # A power-only label or a mixed value/axis string is not whole-item
        # evidence. Requiring the exact item label avoids unit-column inference.
        if not whole_item_marker(claim, item):
            continue
        axis = claim.field.removesuffix("_power") + "_axis"
        paired = [c for c in claims if c.eye == claim.eye and c.field == axis and c.role == "summary"]
        if any(not no_axis_evidence(c) for c in paired):
            continue
        # Contradictory power evidence on this same page cannot authorize an
        # inferred pair state, even if one of the other declarations has dashes.
        powers = [c for c in claims if c is not claim and c.eye == claim.eye and c.field == claim.field and c.role == "summary"]
        if any(c.raw_value is not None or c.missing not in {"not_measured", "not_found", "not_reported"}
               or c.missing == "not_measured" and not whole_item_marker(c, item) for c in powers):
            continue
        if any(c.eye == claim.eye and c.field == axis for c in additions):
            continue
        bbox = claim.bbox
        if bbox is not None and (bbox.x1 <= bbox.x0 or bbox.y1 <= bbox.y0):
            bbox = None
        additions.append(FieldClaim(field=axis, eye=claim.eye, raw_value=None, unit="degree",
            raw_label=claim.raw_label, raw_text=claim.raw_text, bbox=bbox, role="summary",
            missing="not_measured", state_evidence="explicit_marker",
            flags=[*claim.flags, "paired_tk_item_not_measured_v1", "derived_not_model_axis",
                   "shared_item_marker_bbox" if bbox is not None else "source_marker_bbox_unavailable"]))
    return [*claims, *additions]


def parse_page(page: PageInput, output: ProviderOutput, *, profile_id: str,
               model_id: str = "", family: str = "", date_order: str = "mdy") -> ParsedPage:
    """Parse a page without consulting another provider's answers."""
    provider_id = str(output.metadata.get("provider_id", profile_id))
    issues = list(output.issues)
    report_type = "unknown"
    iol_calculations = []
    if output.kind == "claims":
        if output.supported is not True:
            return ParsedPage(supported=False, issues=issues)
        claims = [(claim, None, None) for claim in _paired_tk_item_states(page, output)]
        report_type = str(output.metadata.get("report_type", "structured"))
        from .iol_parsing import parse_iol_tables
        iol_calculations, iol_issues = parse_iol_tables(page, output, profile_id=profile_id, model_id=model_id, family=family)
        issues.extend(iol_issues)
    else:
        rows = _rows(output.spans)
        texts = [row.string()[0] for row in rows]
        compact = re.sub(r"\s+", "", " ".join(texts)).lower()
        supported = "iolmaster700" in compact or ("zeiss" in compact and (("biometricvalues" in compact or "measurements" in compact) and ("k1" in compact or "al:" in compact) or "iolcalculation" in compact))
        if output.supported is False or not supported:
            return ParsedPage(supported=False, issues=issues)
        # Headings precede the repeated generic Biometric values section.
        headings = re.sub(r"\s+", "", " ".join(row.string()[0] for row in rows if row.y < .4)).lower()
        report_type = "calculation" if "iolcalculation" in headings else "analyze" if "analyze" in headings else "reference" if "referenceimage" in headings else "biometric"
        from .iol_parsing import parse_iol_tables, without_iol_table_rows
        iol_calculations, iol_issues = parse_iol_tables(page, output, profile_id=profile_id, model_id=model_id, family=family)
        issues.extend(iol_issues)
        if iol_calculations:
            rows = _rows(without_iol_table_rows(output.spans, iol_calculations))
        claims, parse_issues = _spans_claims(rows, report_type)
        claims = _recover_tk_markers(page, claims)
        issues.extend(parse_issues)
    observations = []
    for claim, confidence, scale in claims:
        observation, problems = _normalize(claim, page, profile_id=profile_id,
                                            model_id=model_id, family=family,
                                            date_order=date_order, provider_id=provider_id,
                                            confidence=confidence, confidence_scale=scale)
        if observation is not None:
            observations.append(observation)
        issues.extend(problems)
    return ParsedPage(supported=True, report_type=report_type, observations=observations, issues=issues,
                      iol_calculations=iol_calculations)

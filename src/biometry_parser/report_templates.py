"""Report identity and layout registry, separate from OCR model selection.

Only IOLMaster 700 is built in. A new manufacturer's matcher does not silently
reuse the ZEISS parser: a template must supply its own parser to be supported.
Anchor geometry is descriptive evidence, never an unconditional crop or eye split.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Sequence

from .models import ParsedPage, Span


@dataclass(frozen=True)
class ReportTemplate:
    id: str
    manufacturer: str
    device: str
    matcher: Callable[[str], bool]
    parser: Callable | None = None
    version: str = "1"


_TEMPLATES: dict[str, ReportTemplate] = {}


def register_report_template(template: ReportTemplate, *, replace: bool = False) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", template.id) or not callable(template.matcher):
        raise ValueError("a report template needs a stable id and matcher")
    if template.parser is not None and not callable(template.parser):
        raise ValueError("a report parser must be callable")
    if template.id in _TEMPLATES and not replace:
        raise ValueError("report template is already registered")
    _TEMPLATES[template.id] = template


def list_report_templates() -> list[dict]:
    return [{"id": template.id, "manufacturer": template.manufacturer, "device": template.device,
             "version": template.version, "supported": template.parser is not None}
            for template in _TEMPLATES.values()]


def _zeiss_parser(*args, **kwargs):
    from .parsing import parse_page
    return parse_page(*args, **kwargs)


register_report_template(ReportTemplate("zeiss_iolmaster_700", "ZEISS", "IOLMaster 700",
                                      lambda text: bool(re.search(r"iolmaster\s*700(?!\d)", text, re.I)),
                                      _zeiss_parser, "anchors-1"))


def inspect_report(spans: Sequence[Span]) -> dict:
    text = " ".join(span.text for span in spans if span.text.strip())
    compact = re.sub(r"\s+", "", text).lower()
    matched = [template for template in _TEMPLATES.values() if template.matcher(text)]
    # Ambiguous matches are not resolved by registration order.
    template = matched[0] if len(matched) == 1 else None
    upper = " ".join(span.text for span in spans if span.bbox.y0 < .4)
    heading = re.sub(r"\s+", "", upper).lower()
    iol_columns = [span for span in spans if re.search(r"\bIOL\b", span.text, re.I)]
    ref_columns = [span for span in spans if re.search(r"\bRef\b|\bRes\.?\s*SE\b", span.text, re.I)]
    table_anchor = any(abs((left.bbox.y0 + left.bbox.y1) - (right.bbox.y0 + right.bbox.y1)) <= .04
                       and abs(left.bbox.x0 - right.bbox.x0) <= .25
                       for left in iol_columns for right in ref_columns)
    calculation = bool(re.search(r"iol\s*calculat", upper, re.I)) or table_anchor
    page_kind = ("iol_calculation" if calculation else "analyze" if "analyze" in heading else
                 "reference_image" if "referenceimage" in heading else
                 "biometry" if template is not None else "unknown")
    measurement_anchors = sorted({match.group(0).upper().rstrip(":") for match in
        re.finditer(r"(?<![A-Za-z])(?:AL|ACD|LT|CCT|WTW|K[12]|TK[12]|Pupil)\b", text, re.I)})
    calculation_anchors = sorted({match.group(0).lower() for match in
        re.finditer(r"(?<![A-Za-z])(?:IOL|Ref\.?|Res\.?\s*SE|Target\s*ref\.?|SRK/T|Barrett|Haigis|Hoffer)\b", text, re.I)})
    eye_headers = {}
    for eye in ("OD", "OS"):
        candidates = [span for span in spans if re.fullmatch(rf"\s*{eye}\s*:?\s*(?:\(?\s*(?:right|left)\s*\)?)?\s*", span.text, re.I)]
        if candidates:
            # Major page headers precede the measurement rows. Retain the box
            # rather than imposing an OD=left / OS=right coordinate assumption.
            chosen = min(candidates, key=lambda item: item.bbox.y0)
            eye_headers[eye] = chosen.bbox.model_dump()
    eye_order = "unresolved"
    aligned = False
    if len(eye_headers) == 2:
        od, os = eye_headers["OD"], eye_headers["OS"]
        od_x, os_x = (od["x0"] + od["x1"]) / 2, (os["x0"] + os["x1"]) / 2
        aligned = abs((od["y0"] + od["y1"]) - (os["y0"] + os["y1"])) <= .08 and abs(od_x - os_x) >= .15
        if aligned:
            eye_order = "OD_left_OS_right" if od_x < os_x else "OS_left_OD_right"
    elif len(eye_headers) == 1:
        eye_order = "single_" + next(iter(eye_headers))
        aligned = True
    reasons = []
    if len(matched) > 1:
        reasons.append("ambiguous_report_template")
    elif template is None:
        reasons.append("unsupported_report_template")
    elif template.parser is None:
        reasons.append("report_parser_not_registered")
    if len(text) < 60:
        reasons.append("native_text_too_short")
    if "\ufffd" in text or "\x00" in text:
        reasons.append("native_text_corrupt")
    identity = "patientid" in compact and "measurement" in compact
    if not identity:
        reasons.append("native_identity_anchors_missing")
    payload = len(measurement_anchors) >= 2 or (calculation and len(calculation_anchors) >= 2)
    if not payload:
        reasons.append("native_measurement_anchors_missing")
    supported = template is not None and template.parser is not None
    explicit_device = re.search(r"iolmaster\s*(\d+)(?!\d)", text, re.I)
    named_other_device = None
    if template is None:
        for pattern, name in ((r"\bLENSTAR(?:\s+LS\s*900)?\b", "LENSTAR"),
                              (r"\bOA[-\s]?2000\b", "OA-2000"),
                              (r"\bAL[-\s]?SCAN\b", "AL-Scan"),
                              (r"\bARGOS\s+(?:optical\s+)?biometer\b", "ARGOS")):
            if re.search(pattern, text, re.I):
                named_other_device = name
                break
    other_device = template is None and bool(named_other_device or explicit_device and explicit_device.group(1) != "700")
    return {"template_id": template.id if template else None,
            "template_version": template.version if template else None,
            "manufacturer": template.manufacturer if template else "ZEISS" if "zeiss" in compact or "iolmaster" in compact else None,
            "device": template.device if template else "IOLMaster " + explicit_device.group(1) if explicit_device else named_other_device,
            "supported": supported, "explicit_unsupported_device": other_device,
            "brand_status": "registered_supported" if supported else "unsupported_device" if other_device else "unconfirmed",
            "page_kind": page_kind,
            "native_quality": {"usable": supported and not reasons, "reason_codes": reasons,
                               "text_character_count": len(text), "positioned_span_count": len(spans)},
            "layout": {"validated": supported and aligned and payload,
                       "eye_order": eye_order, "eye_headers": eye_headers,
                       "calculation_table_anchors_validated": table_anchor,
                       "measurement_anchors": measurement_anchors, "calculation_anchors": calculation_anchors,
                       "coordinate_policy": "observed_original_page_anchors", "automatic_crop_applied": False}}


def parse_registered_page(page, output, *, inspection: dict | None = None, **kwargs) -> ParsedPage:
    if output.kind == "claims":
        # A structured provider declares report support in its checked contract.
        from .parsing import parse_page
        return parse_page(page, output, **kwargs)
    inspection = inspection if inspection is not None else inspect_report(output.spans)
    template = _TEMPLATES.get(inspection.get("template_id"))
    if template is None or template.parser is None:
        return ParsedPage(supported=False)
    return template.parser(page, output, **kwargs)

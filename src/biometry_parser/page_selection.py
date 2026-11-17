"""Inspectable, opt-in page selection without running an OCR model.

Native text pages can be selected by semantic summary coverage. Unknown pages
are retained. Raster-only pages are skipped only when their input pixels are
identical at the requested recognition DPI; visual similarity is not equality.
"""
from __future__ import annotations

from decimal import Decimal
import hashlib
from pathlib import Path
import time

from PIL import Image, ImageOps

from .documents import _native_spans, display_rgb, native_text_usable
from .models import PageInput, ProviderOutput
from .parsing import parse_page
from .report_templates import inspect_report


def _claim_key(observation):
    value = observation.value
    if isinstance(value, Decimal):
        value = value % 180 if observation.field.endswith("_axis") else value
        value = str(value.normalize())
    return observation.field, observation.eye, observation.status, value, observation.unit, tuple(sorted(observation.flags))


def _pixels(image) -> str:
    rgb = display_rgb(image)
    try:
        digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode())
        # Bound temporary allocations to a strip rather than duplicating a
        # complete multi-megapixel page into one Python bytes object.
        for y in range(0, rgb.height, 128):
            strip = rgb.crop((0, y, rgb.width, min(y + 128, rgb.height)))
            try:
                digest.update(strip.tobytes())
            finally:
                strip.close()
        return digest.hexdigest()
    finally:
        rgb.close()


def select_native_coverage(candidates: dict[int, set[tuple]]) -> list[int]:
    """Cover all distinct claims, including conflicting values and states.

    This is deliberately more conservative than covering only field names:
    two pages with different AL values must both survive the selection.
    """
    uncovered = set().union(*candidates.values()) if candidates else set()
    selected = []
    while uncovered:
        number = max(candidates, key=lambda n: (len(candidates[n] & uncovered), len(candidates[n]), -n))
        gained = candidates[number] & uncovered
        if not gained:
            break
        selected.append(number)
        uncovered -= gained
    return sorted(selected)


def select_biometry_pages(candidates: dict[int, set[tuple]]) -> dict:
    """Start with the most informative eye summary, then cover distinct facts.

    Count field/eye slots before considering CCT: a tiny CCT-only page must
    never outrank a complete bilateral summary. CCT breaks coverage ties.
    The coverage target includes identities and contradictory values, so the
    primary page cannot hide a different patient/date or disagreeing reading.
    This ranks observed native evidence; it does not prove unprinted fields
    absent and cannot rank image content without a recognizer.
    """
    ranking = []
    for number, keys in candidates.items():
        ocular = {(key[0], key[1]) for key in keys if key[1] in {"OD", "OS"}
                  and (key[2] == "present" and key[3] is not None
                       or key[2] in {"not_measured", "blank", "absent"})}
        by_eye = {eye: {field for field, side in ocular if side == eye} for eye in ("OD", "OS")}
        ranking.append({"page_number": number, "answered_ocular_fields": len(ocular),
                        "bilateral_fields": len(by_eye["OD"] & by_eye["OS"]),
                        "cct_eyes": sum(("cct", eye) in ocular for eye in ("OD", "OS")),
                        "identity_fields": len({key[0] for key in keys if key[1] is None}),
                        "distinct_claim_count": len(keys)})
    ranking.sort(key=lambda row: (-row["answered_ocular_fields"], -row["bilateral_fields"],
                                  -row["cct_eyes"], -row["identity_fields"], -row["distinct_claim_count"],
                                  row["page_number"]))
    primary = next((row["page_number"] for row in ranking if row["answered_ocular_fields"]), None)
    uncovered = set().union(*candidates.values()) if candidates else set()
    selected = [primary] if primary is not None else []
    if primary is not None:
        uncovered -= candidates[primary]
    while uncovered:
        number = max(candidates, key=lambda n: (len(candidates[n] & uncovered), len(candidates[n]), -n))
        gained = candidates[number] & uncovered
        if not gained:
            break
        selected.append(number)
        uncovered -= gained
    return {"primary_biometry_page": primary, "biometry_pages": selected,
            "fallback_biometry_pages": [n for n in selected if n != primary],
            "native_biometry_ranking": ranking}


def plan_pages(source: str | Path, *, strategy: str = "auto", pages: list[int] | None = None,
               dpi: int = 300, max_page_pixels: int = 20_000_000, max_pages: int = 50,
               date_order: str = "mdy") -> dict:
    """Return original selected page numbers and an explanation for every page.

    ``auto`` never calls a local or remote recognizer. Native semantic selection
    uses all distinct summary claims, while retaining unsupported or incompletely
    identified pages. This saves recognition work; it is not a quality guarantee.
    ``biometry_first`` starts with the richest native eye summary, adds pages
    needed for distinct missing/conflicting facts, and separately retains IOL
    calculation pages. Raster-only pages conservatively require recognition.
    ``all`` is the unchanged complete-document comparison mode.
    """
    started = time.perf_counter()
    path = Path(source).resolve(strict=True)
    if strategy not in {"auto", "biometry_first", "all", "current", "custom"}:
        raise ValueError("unknown page selection strategy")
    if not 72 <= dpi <= 600 or max_page_pixels < 1 or max_pages < 1:
        raise ValueError("invalid page selection limits")
    details, candidates, retained = [], {}, set()
    biometry_candidates, iol_pages, raster_pages = {}, set(), set()
    pixel_pages = {}

    def inspect_native(page, number):
        width, height = page.get_size()
        spans = _native_spans(page)
        sample = PageInput(page_number=number, image_path="", width=max(1, round(width)),
                           height=max(1, round(height)), native_spans=spans, source_path=str(path))
        report = inspect_report(spans)
        if not native_text_usable(sample):
            if report["supported"] and report["page_kind"] == "iol_calculation":
                retained.add(number)
                iol_pages.add(number)
                return {"page_number": number, "reason": "native_calculation_content_retained", "coverage_count": 0,
                        "iol_table_count": 0, "report_template": report}
            return None
        parsed = parse_page(sample, ProviderOutput(kind="spans", spans=spans), profile_id="page_planner",
                            model_id="pdfium-text", family="native_pdf", date_order=date_order)
        summaries = [item for item in parsed.observations if item.role == "summary"]
        # Repeated biometry is not the whole report: lens, formula, constant and
        # candidate IOL-power tables can differ on every calculation page. Keep
        # these pages even if all 37 scalar summary fields are already covered.
        # Text equality is not enough to deduplicate a gray/highlighted choice.
        is_iol_page = bool(parsed.iol_calculations or report["page_kind"] == "iol_calculation")
        if is_iol_page:
            retained.add(number)
            iol_pages.add(number)
            if strategy == "auto":
                return {"page_number": number, "reason": "native_calculation_content_retained",
                        "coverage_count": len(summaries), "iol_table_count": len(parsed.iol_calculations), "report_template": report}
        extra = {"iol_table_count": len(parsed.iol_calculations)} if is_iol_page else {}
        identity = {item.field for item in summaries if item.status == "present" and item.value is not None}
        if parsed.issues or any(item.status == "unreadable" for item in summaries):
            retained.add(number)
            return {"page_number": number, "reason": "native_uncertain_evidence_requires_review", "coverage_count": len(summaries),
                    "report_template": report, **extra}
        if not parsed.supported or not {"patient_id", "measurement_date"} <= identity:
            retained.add(number)
            return {"page_number": number, "reason": "native_identity_or_layout_incomplete", "coverage_count": 0,
                    "report_template": report, **extra}
        keys = {_claim_key(item) for item in summaries}
        if not keys:
            return None
        candidates[number] = keys
        biometry_candidates[number] = keys
        return {"page_number": number, "reason": "native_calculation_content_retained" if is_iol_page else "native_summary_candidate",
                "coverage_count": len(keys), "report_template": report, **extra}

    def raster_detail(number, image):
        digest = _pixels(image)
        previous = pixel_pages.get(digest)
        if previous is not None:
            return {"page_number": number, "reason": "identical_input_pixels", "duplicate_of": previous,
                    "coverage_count": 0, "pixel_sha256": digest}
        pixel_pages[digest] = number
        retained.add(number)
        raster_pages.add(number)
        return {"page_number": number, "reason": "raster_content_requires_recognition", "coverage_count": 0,
                "pixel_sha256": digest}

    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium
        with pdfium.PdfDocument(path) as document:
            count = len(document)
            if not 1 <= count <= max_pages:
                raise ValueError("document exceeds max_pages")
            if strategy in {"auto", "biometry_first"}:
                for index in range(count):
                    page = document[index]
                    try:
                        native = inspect_native(page, index + 1)
                        if native is not None:
                            details.append(native)
                            continue
                        width, height = page.get_size()
                        scale = dpi / 72
                        if width * height * scale * scale > max_page_pixels:
                            raise ValueError("page exceeds max_page_pixels")
                        bitmap = page.render(scale=scale)
                        try:
                            image = bitmap.to_pil()
                            try:
                                details.append(raster_detail(index + 1, image))
                            finally:
                                image.close()
                        finally:
                            bitmap.close()
                    finally:
                        page.close()
    else:
        with Image.open(path) as original:
            count = getattr(original, "n_frames", 1)
            if not 1 <= count <= max_pages:
                raise ValueError("document exceeds max_pages")
            if strategy in {"auto", "biometry_first"}:
                for index in range(count):
                    original.seek(index)
                    if original.width * original.height > max_page_pixels:
                        raise ValueError("page exceeds max_page_pixels")
                    image = ImageOps.exif_transpose(original)
                    try:
                        details.append(raster_detail(index + 1, image))
                    finally:
                        if image is not original:
                            image.close()

    if strategy in {"current", "custom"}:
        if (not isinstance(pages, list) or not pages or any(type(number) is not int or not 1 <= number <= count for number in pages)
                or len(set(pages)) != len(pages) or (strategy == "current" and len(pages) != 1)):
            raise ValueError("invalid explicit page numbers")
        selected = sorted(pages)
        details = [{"page_number": number, "reason": "explicit_page_selection", "coverage_count": 0} for number in range(1, count + 1)]
    elif strategy == "all":
        if pages is not None:
            raise ValueError("all strategy does not accept a page subset")
        selected = list(range(1, count + 1))
        details = [{"page_number": number, "reason": "complete_document", "coverage_count": 0} for number in selected]
    elif strategy == "biometry_first":
        if pages is not None:
            raise ValueError("biometry_first strategy does not accept explicit pages")
        biometry_selection = select_biometry_pages(biometry_candidates)
        chosen = set(biometry_selection["biometry_pages"])
        selected = sorted(retained | chosen)
        for detail in details:
            number = detail["page_number"]
            roles = []
            if number in iol_pages:
                roles.append("iol_calculation")
            if number in chosen:
                primary = number == biometry_selection["primary_biometry_page"]
                roles.append("biometry_primary" if primary else "biometry_fallback")
                if number not in iol_pages:
                    detail["reason"] = "native_primary_biometry_page" if primary else "native_biometry_fallback_for_distinct_claims"
            elif number in biometry_candidates and number not in iol_pages:
                detail["reason"] = "native_biometry_already_covered"
            if number in raster_pages:
                roles.append("raster_requires_recognition")
            elif (number in retained and number not in iol_pages or detail["reason"] in {
                    "native_uncertain_evidence_requires_review", "native_identity_or_layout_incomplete"}):
                roles.append("required_review")
            detail["selection_roles"] = roles
    else:
        if pages is not None:
            raise ValueError("auto strategy does not accept explicit pages")
        selected = sorted(retained | set(select_native_coverage(candidates)))
        for detail in details:
            if detail["page_number"] in candidates:
                detail["reason"] = "native_distinct_summary_coverage" if detail["page_number"] in selected else "native_summary_already_covered"
    for detail in details:
        detail["selected"] = detail["page_number"] in selected
    source_digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            source_digest.update(chunk)
    selection_metadata = {}
    if strategy == "biometry_first":
        limitations = ["raster_pages_cannot_be_ranked_without_recognition"] if raster_pages else []
        if biometry_selection["primary_biometry_page"] is not None:
            limitations.append("native_selection_does_not_validate_ocr_field_completeness")
        selection_metadata = {**biometry_selection, "iol_pages": sorted(iol_pages),
                              "raster_pages_requiring_recognition": sorted(raster_pages),
                              "biometry_selection_limitations": limitations,
                              "automatic_retry_for_selected_ocr_missing_fields": False,
                              "biometry_coverage_scope": "all_distinct_reliable_native_summary_claims_not_all_possible_report_fields"}
    return {"schema_version": "1.0", "strategy": strategy, "selected_pages": selected,
            "selection_policy": "native_biometry_first_preserve_iol_v1" if strategy == "biometry_first" else "native_summary_cover_preserve_iol_calculations_v2",
            "source_sha256": source_digest.hexdigest(),
            "source_page_count": count, "skipped_pages": [number for number in range(1, count + 1) if number not in selected],
            "page_details": details, "reasons": sorted({item["reason"] for item in details}),
            "coverage": {"native_candidate_pages": len(candidates),
                         "native_iol_calculation_pages_retained": sum(item["reason"] == "native_calculation_content_retained" for item in details),
                         "native_iol_table_count": sum(item.get("iol_table_count", 0) for item in details),
                         "distinct_native_summary_claims": len(set().union(*candidates.values())) if candidates else 0},
            "recognition_calls_made": 0, "inference_used": False,
            "raster_comparison_dpi": dpi, "planning_seconds": time.perf_counter() - started,
            "scope_warning": "Selected-page accuracy covers this plan only; use all pages to audit cross-page OCR disagreements.",
            **selection_metadata}

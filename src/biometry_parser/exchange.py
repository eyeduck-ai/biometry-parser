"""Small canonical exchange documents without duplicated recognition evidence.

This is a lossy transport profile of ExtractionResult 1.2, not a source archive
and not a second extraction or verification pass.
"""
from __future__ import annotations

from .models import ExtractionResult, SCHEMA_VERSION

_METADATA_KEYS = {
    "biometry_clip", "merge_policy_version", "source_state_counts",
    "processing_status", "source_page_count", "selected_pages",
    "page_strategy", "page_selection_explicit",
}
_SCALAR_FIELDS = (
    "al", "acd", "lt", "cct", "wtw", "mean_k", "corneal_astigmatism",
    "corneal_astigmatism_axis", "pupil",
)
_K_FIELDS = ("k1", "k2", "tk1", "tk2")
_OMITTED = (
    "runs", "field_candidates", "bbox", "raw_text",
    "selection_evidence", "highlight_evidence", "execution_metadata",
)


def to_exchange_document(result: ExtractionResult) -> dict:
    """Return compact JSON-ready canonical data, without changing its meaning.

    Keep every final scalar, source state, unit, verification state and IOL
    calculation/table identity. Omit observation arrays, duplicated runs, raw
    text and coordinates before serialization, so large evidence is not copied
    into a temporary JSON document. No providers are invoked and no source hash,
    page, observation, accepted value or verification is manufactured.

    The result remains loadable with ExtractionResult.model_validate_json().
    Namespace metadata.biometry_clip may carry the caller's portable Clip case
    and per-eye settings. Its content is forwarded, not interpreted or trusted.
    """
    if not isinstance(result, ExtractionResult):
        raise TypeError("result must be an ExtractionResult")
    if result.schema_version != SCHEMA_VERSION:
        raise ValueError("compact exchange requires ExtractionResult schema 1.2")

    eye_exclude = {name: {"candidates"} for name in _SCALAR_FIELDS}
    eye_exclude.update({
        name: {"power": {"candidates"}, "axis": {"candidates"}}
        for name in _K_FIELDS
    })
    exclude = {
        "runs": True,
        "metadata": True,
        "patient_id": {"candidates"},
        "gender": {"candidates"},
        "measurement_date": {"candidates"},
        "eyes": {"__all__": eye_exclude},
        "iol_calculations": {"__all__": {
            "bbox": True,
            "raw_text": True,
            "constants": {"__all__": {"raw_text", "bbox"}},
            "rows": {"__all__": {
                "bbox", "raw_text", "selection_evidence", "highlight_evidence",
            }},
        }},
    }
    document = result.model_dump(mode="json", exclude=exclude)
    metadata = result.model_dump(
        mode="json", include={"metadata": _METADATA_KEYS},
    )["metadata"]
    metadata["biometry_parser_exchange"] = {
        "profile": "compact",
        "version": "1.0",
        "omitted": list(_OMITTED),
    }
    document["metadata"] = metadata
    return document

"""Shared, inspectable cross-page merging of one field and eye.

Only actual reported values and explicit not-measured markers can contradict
one another. Unreadable, unlisted and absent page evidence is retained for
inspection but cannot erase a decisive observation. No voting is performed.
This resolves candidate evidence; independent-provider verification and source
identity checks remain the responsibility of the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from .models import EYE_FIELDS, UNITS, Observation

POLICY_VERSION = "cross_page_v3"
_ABSENCE_PROOF = {"human_review", "verified_template"}
_BLANK_PROOF = {"human_review", "verified_template", "confirmed_empty_region", "provider_assertion"}


def _numeric(value):
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except (InvalidOperation, TypeError, ValueError):
        return None


def comparison_key(field: str, status: str, value: Any, unit: str | None):
    """Compare normalized values while preserving the original representation."""
    if status == "not_reported":
        status = "not_found"
    if status != "present":
        return status, None, None
    if value is None:
        return "unreadable", None, None
    if field in EYE_FIELDS:
        number = _numeric(value)
        if number is None or (field.endswith("_axis") and not 0 <= number <= 180):
            return "unreadable", None, None
        value = number % 180 if field.endswith("_axis") else number
    else:
        value = str(value)
    return status, value, unit or UNITS.get(field)


def _effective_status(observation):
    if observation.status == "not_reported":
        return "not_found"
    if observation.status == "absent" and observation.state_evidence not in _ABSENCE_PROOF:
        return "not_found"
    if observation.status == "blank" and observation.state_evidence not in _BLANK_PROOF:
        return "unreadable"
    if observation.status == "present":
        if observation.value is None:
            return "unreadable"
        return comparison_key(observation.field, observation.status, observation.value, observation.unit)[0]
    return observation.status


def _json_value(value):
    return float(value) if isinstance(value, Decimal) else value


@dataclass
class MergeResolution:
    field: str
    eye: str | None
    status: str
    value: Any
    unit: str | None
    reason_code: str
    contributing_pages: list[int] = dataclass_field(default_factory=list)
    ignored_pages: list[int] = dataclass_field(default_factory=list)
    missing_pages: list[int] = dataclass_field(default_factory=list)
    conflicting_pages: list[int] = dataclass_field(default_factory=list)
    pages: list[dict] = dataclass_field(default_factory=list)
    policy_version: str = POLICY_VERSION

    def to_dict(self) -> dict:
        return {"policy_version": self.policy_version, "field": self.field, "eye": self.eye,
                "status": self.status, "data_status": self.status, "value": _json_value(self.value), "unit": self.unit,
                "reason_code": self.reason_code, "contributing_pages": list(self.contributing_pages),
                "ignored_pages": list(self.ignored_pages), "missing_pages": list(self.missing_pages),
                "conflicting_pages": list(self.conflicting_pages), "pages": self.pages}


def _resolve(observations: list[Observation], field: str, eye: str | None, missing_pages: bool = False):
    summaries = [o for o in observations if o.role == "summary"]
    decisive = [o for o in summaries if _effective_status(o) in {"present", "not_measured"}]
    keys = {comparison_key(field, _effective_status(o), o.value, o.unit) for o in decisive}
    unit = UNITS.get(field)
    if len(keys) > 1:
        reason = "value_vs_not_measured" if any(key[0] == "not_measured" for key in keys) else "different_values"
        return MergeResolution(field, eye, "conflict", None, unit, reason), decisive
    unknown = len(summaries) > len(decisive) or missing_pages
    if decisive:
        first = decisive[0]
        status = _effective_status(first)
        reason = ("value_retained_over_missing" if unknown else "consistent_value") if status == "present" else (
                  "not_measured_retained_over_missing" if unknown else "explicit_not_measured")
        return MergeResolution(field, eye, status, first.value if status == "present" else None,
                               first.unit or unit, reason), decisive
    if any(_effective_status(o) == "unreadable" for o in summaries):
        return MergeResolution(field, eye, "unreadable", None, unit, "unreadable_without_decisive_evidence"), decisive
    known_empty = [o for o in summaries if _effective_status(o) in {"blank", "absent"}]
    if known_empty and len(known_empty) == len(summaries) and not missing_pages:
        status = "blank" if any(_effective_status(o) == "blank" for o in known_empty) else "absent"
        reason = "blank_confirmed_in_scope" if status == "blank" else "absence_confirmed_in_scope"
        return MergeResolution(field, eye, status, None, unit, reason), known_empty
    reason = "empty_evidence_incomplete_scope" if known_empty else "not_found_only" if summaries else "no_summary_evidence"
    return MergeResolution(field, eye, "not_found", None, unit, reason), decisive


def merge_observations(observations: Iterable[Observation], field: str, eye: str | None,
                       *, pages: Iterable[int] | None = None) -> MergeResolution:
    """Return a merged candidate and a complete per-page explanation.

    ``pages`` is the recognizer's selected original-page scope. Without it, all
    observed run pages are included. Individual measurements remain in the
    explanation but never participate in summary merging. Candidate indices
    match the input's filtered field/eye list, including individual evidence.
    """
    all_observations = list(observations)
    candidates = [o for o in all_observations if o.field == field and o.eye == eye]
    page_numbers = sorted(set(pages or ()) | {o.page_number for o in all_observations})
    missing = [n for n in page_numbers if not any(o.page_number == n and o.role == "summary" for o in candidates)]
    merged, decisive = _resolve(candidates, field, eye, bool(missing))
    decisive_ids = {id(o) for o in decisive}
    is_conflict = merged.status == "conflict"
    for number in page_numbers:
        indexed = [(index, o) for index, o in enumerate(candidates) if o.page_number == number]
        source = [o for _, o in indexed]
        page_resolution, page_decisive = _resolve(source, field, eye)
        if page_decisive:
            contributes = any(id(o) in decisive_ids for o in page_decisive)
            decision = "conflicting" if is_conflict and contributes else "contributing" if contributes else "ignored"
            reason = "conflicting_decisive_evidence" if decision == "conflicting" else (
                "supports_merged_value" if merged.status == "present" else "supports_not_measured" if merged.status == "not_measured"
                else "supports_empty_state") if contributes else (
                    "empty_page_does_not_override_value" if merged.status in {"present", "not_measured", "conflict"}
                    else "empty_evidence_incomplete_scope")
        elif not source:
            decision, reason = "no_evidence", "no_observation"
        else:
            decision = "ignored"
            reason = ("ignored_unreadable" if page_resolution.status == "unreadable" else
                      "ignored_not_found" if any(o.role == "summary" for o in source) else "ignored_individual")
        evidence = []
        for index, observation in indexed:
            state = _effective_status(observation)
            if observation.role != "summary":
                source_decision, source_reason = "ignored", "ignored_individual"
            elif id(observation) in decisive_ids:
                source_decision, source_reason = decision, reason
            else:
                source_decision = "ignored"
                source_reason = "ignored_unreadable" if state == "unreadable" else (
                    "empty_page_does_not_override_value" if state in {"blank", "absent"} and merged.status in {"present", "not_measured", "conflict"}
                    else "empty_evidence_incomplete_scope" if state in {"blank", "absent"} else "ignored_not_found")
            evidence.append({**observation.model_dump(mode="json"), "candidate_index": index,
                             "effective_status": state, "decision": source_decision, "reason_code": source_reason})
        merged.pages.append({"page_number": number, "status": page_resolution.status, "data_status": page_resolution.status,
                             "value": _json_value(page_resolution.value), "unit": page_resolution.unit,
                             "decision": decision, "reason_code": reason, "observations": evidence})
    merged.contributing_pages = [p["page_number"] for p in merged.pages if p["decision"] == "contributing"]
    merged.ignored_pages = [p["page_number"] for p in merged.pages if p["decision"] == "ignored"]
    merged.missing_pages = missing
    merged.conflicting_pages = [p["page_number"] for p in merged.pages if p["decision"] == "conflicting"]
    return merged

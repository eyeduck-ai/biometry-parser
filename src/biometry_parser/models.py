"""Public versioned contracts shared by adapters, parsers and callers."""
from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

SCHEMA_VERSION = "1.2"
PROTOCOL_VERSION = "1.0"
Eye = Literal["OD", "OS"]
SourceStatus = Literal["present", "not_found", "absent", "blank", "not_measured", "unreadable", "not_reported"]
MissingStatus = Literal["not_found", "absent", "blank", "not_measured", "unreadable", "not_reported"]
StateEvidence = Literal["legacy_unspecified", "none", "label_only", "provider_assertion", "explicit_marker",
                        "confirmed_empty_region", "human_review", "verified_template"]
FIELDS = ("al", "acd", "lt", "cct", "wtw", "mean_k", "corneal_astigmatism", "corneal_astigmatism_axis", "k1_power", "k1_axis", "k2_power", "k2_axis", "tk1_power", "tk1_axis", "tk2_power", "tk2_axis", "pupil", "patient_id", "gender", "measurement_date")
EYE_FIELDS = FIELDS[:-3]
UNITS = {**dict.fromkeys(("al", "acd", "lt", "wtw", "pupil"), "mm"), "cct": "um", **dict.fromkeys(("mean_k", "corneal_astigmatism", "k1_power", "k2_power", "tk1_power", "tk2_power"), "D"), **dict.fromkeys(("corneal_astigmatism_axis", "k1_axis", "k2_axis", "tk1_axis", "tk2_axis"), "degree")}

class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

class Box(Contract):
    """Normalized coordinates on the displayed original page, top-left origin."""
    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self):
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("box coordinates must be ordered")
        return self

class Span(Contract):
    text: str
    bbox: Box
    confidence: float | None = None
    confidence_scale: str | None = None

class PageInput(Contract):
    page_number: int = Field(ge=1)
    image_path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    native_spans: list[Span] = Field(default_factory=list)
    source_path: str = ""

class Issue(Contract):
    code: str
    message: str
    page_number: int | None = None
    field: str | None = None
    eye: Eye | None = None
    profile_id: str | None = None

class FieldClaim(Contract):
    field: str
    eye: Eye | None = None
    raw_value: str | None = None
    unit: str | None = None
    raw_label: str = ""
    raw_text: str = ""
    bbox: Box | None = None
    role: Literal["summary", "individual", "unknown"] = "summary"
    missing: MissingStatus | None = None
    state_evidence: StateEvidence = "legacy_unspecified"
    flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_new_missing_state(self):
        if self.missing in {"not_found", "absent", "blank"} and self.raw_value is not None and self.raw_value.strip():
            raise ValueError("a non-present source state cannot contain a value")
        return self

class IOLConstantClaim(Contract):
    """A printed lens/formula constant; values may be numeric or 'Default'."""
    name: str
    value: str
    raw_text: str = ""
    bbox: Box | None = None


class IOLRowClaim(Contract):
    """Provider-transcribed row, before deterministic numeric normalization.

    A shaded recommendation is a highlight, not evidence of surgical selection.
    ``selection`` describes only an explicit printed selection indication.
    """
    iol_power: str | None = None
    predicted_refraction: str | None = None
    toric_power: str | None = None
    axis: str | None = None
    row_kind: Literal["candidate", "emmetropia", "unknown"] = "candidate"
    selection: Literal["marked", "not_marked", "unknown"] = "unknown"
    selection_evidence: str = ""
    highlight_status: Literal["highlighted", "not_highlighted", "unknown"] = "unknown"
    highlight_evidence: str = ""
    raw_text: str = ""
    bbox: Box | None = None


class IOLTableClaim(Contract):
    """A single printed calculation table, never merged by lens name alone."""
    eye: Eye | None = None
    lens_model: str | None = None
    lens_type: Literal["toric", "non_toric", "unknown"] = "unknown"
    formula: str | None = None
    keratometry_source: Literal["K", "TK", "unknown"] = "unknown"
    target_refraction: str | None = None
    constants: list[IOLConstantClaim] = Field(default_factory=list)
    rows: list[IOLRowClaim] = Field(default_factory=list)
    raw_text: str = ""
    bbox: Box | None = None


class IOLRow(Contract):
    iol_power: Decimal | None = None
    predicted_refraction: Decimal | None = None
    toric_power: Decimal | None = None
    axis: Decimal | None = None
    row_kind: Literal["candidate", "emmetropia", "unknown"] = "candidate"
    selection: Literal["marked", "not_marked", "unknown"] = "unknown"
    selection_evidence: str = ""
    highlight_status: Literal["highlighted", "not_highlighted", "unknown"] = "unknown"
    highlight_evidence: str = ""
    raw_text: str = ""
    bbox: Box | None = None
    flags: list[str] = Field(default_factory=list)

    @field_serializer("iol_power", "predicted_refraction", "toric_power", "axis", when_used="json")
    def json_number(self, value) -> float | None:
        return float(value) if value is not None else None


class IOLCalculation(Contract):
    """Unverified table-level source evidence, separate from biometric values."""
    table_id: str
    eye: Eye | None = None
    lens_model: str | None = None
    lens_type: Literal["toric", "non_toric", "unknown"] = "unknown"
    formula: str | None = None
    keratometry_source: Literal["K", "TK", "unknown"] = "unknown"
    target_refraction: Decimal | None = None
    constants: list[IOLConstantClaim] = Field(default_factory=list)
    rows: list[IOLRow] = Field(default_factory=list)
    page_number: int = Field(ge=1)
    bbox: Box | None = None
    raw_text: str = ""
    provider_id: str = ""
    profile_id: str = ""
    model_id: str = ""
    model_hash: str | None = None
    family: str = ""
    flags: list[str] = Field(default_factory=list)
    verification_status: Literal["unverified", "cross_checked"] = "unverified"

    @field_serializer("target_refraction", when_used="json")
    def json_number(self, value) -> float | None:
        return float(value) if value is not None else None

class ProviderOutput(Contract):
    kind: Literal["spans", "claims"]
    spans: list[Span] = Field(default_factory=list)
    claims: list[FieldClaim] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    supported: bool | None = None
    iol_calculations: list[IOLTableClaim] = Field(default_factory=list)

class Observation(Contract):
    field: str
    eye: Eye | None = None
    value: str | Decimal | None = None
    unit: str | None = None
    status: SourceStatus = "present"
    page_number: int = Field(ge=1)
    bbox: Box | None = None
    raw_label: str = ""
    raw_text: str = ""
    provider_id: str = ""
    profile_id: str = ""
    model_id: str = ""
    model_hash: str | None = None
    family: str = ""
    independence_group: str | None = None
    role: Literal["summary", "individual", "unknown"] = "summary"
    flags: list[str] = Field(default_factory=list)
    confidence: float | None = None
    confidence_scale: str | None = None
    state_evidence: StateEvidence = "legacy_unspecified"

    @model_validator(mode="after")
    def consistent_new_source_state(self):
        if self.status in {"not_found", "absent", "blank"} and self.value is not None:
            raise ValueError("a non-present source state cannot contain a value")
        return self

    @field_serializer("value", when_used="json")
    def json_value(self, value) -> float | str | None:
        return float(value) if isinstance(value, Decimal) else value

class ParsedPage(Contract):
    supported: bool
    report_type: str = "unknown"
    observations: list[Observation] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    iol_calculations: list[IOLCalculation] = Field(default_factory=list)

class Status(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    NOT_REPORTED = "not_reported"
    NOT_FOUND = "not_found"
    ABSENT = "absent"
    BLANK = "blank"
    NOT_MEASURED = "not_measured"
    UNREADABLE = "unreadable"
    CONFLICT = "conflict"
    ERROR = "error"


class DataStatus(str, Enum):
    """Source facts, independent of execution success and verification."""
    PRESENT = "present"
    NOT_FOUND = "not_found"
    ABSENT = "absent"
    BLANK = "blank"
    NOT_MEASURED = "not_measured"
    UNREADABLE = "unreadable"
    CONFLICT = "conflict"

class FieldResult(Contract):
    value: str | Decimal | None = None
    unit: str | None = None
    status: Status = Status.NOT_REPORTED
    data_status: DataStatus = DataStatus.NOT_FOUND
    candidate_value: str | Decimal | None = None
    verification_status: Literal["unverified", "cross_checked"] = "unverified"
    candidates: list[Observation] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def load_legacy_source_information(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        status = data.get("status")
        if isinstance(status, Status):
            status = status.value
        if "data_status" not in data:
            if status == "verified":
                data["data_status"] = "present"
            elif status in {"not_found", "absent", "blank", "not_measured", "unreadable", "conflict"}:
                data["data_status"] = status
            else:
                data["data_status"] = "not_found"
        if "candidate_value" not in data and data["data_status"] in {"present", DataStatus.PRESENT}:
            data["candidate_value"] = data.get("value")
        if "verification_status" not in data and status == "verified":
            data["verification_status"] = "cross_checked"
        return data

    @model_validator(mode="after")
    def consistent_candidate_value(self):
        if self.data_status != DataStatus.PRESENT and self.candidate_value is not None:
            raise ValueError("only a present data state has a candidate value")
        if self.data_status != DataStatus.PRESENT and self.value is not None:
            raise ValueError("a non-present data state cannot contain an accepted value")
        if self.data_status == DataStatus.PRESENT and self.candidate_value is None:
            raise ValueError("a present data state requires a candidate value")
        if self.status != Status.VERIFIED and self.value is not None:
            raise ValueError("only verified fields have an accepted value")
        if self.status == Status.VERIFIED:
            if self.data_status != DataStatus.PRESENT or self.value is None or self.candidate_value != self.value:
                raise ValueError("verified values require matching present source and candidate values")
            if self.verification_status != "cross_checked":
                raise ValueError("verified status requires cross-checked verification")
        if self.verification_status == "cross_checked":
            accepted = {DataStatus.PRESENT: Status.VERIFIED, DataStatus.NOT_MEASURED: Status.NOT_MEASURED,
                        DataStatus.BLANK: Status.BLANK, DataStatus.ABSENT: Status.ABSENT}
            if accepted.get(self.data_status) != self.status:
                raise ValueError("cross-checked verification requires an accepted source fact")
        return self

    @field_serializer("value", "candidate_value", when_used="json")
    def json_value(self, value) -> float | str | None:
        return float(value) if isinstance(value, Decimal) else value

class Keratometry(Contract):
    power: FieldResult = Field(default_factory=lambda: FieldResult(unit="D"))
    axis: FieldResult = Field(default_factory=lambda: FieldResult(unit="degree"))

class EyeResult(Contract):
    al: FieldResult = Field(default_factory=lambda: FieldResult(unit="mm"))
    acd: FieldResult = Field(default_factory=lambda: FieldResult(unit="mm"))
    lt: FieldResult = Field(default_factory=lambda: FieldResult(unit="mm"))
    cct: FieldResult = Field(default_factory=lambda: FieldResult(unit="um"))
    wtw: FieldResult = Field(default_factory=lambda: FieldResult(unit="mm"))
    mean_k: FieldResult = Field(default_factory=lambda: FieldResult(unit="D"))
    corneal_astigmatism: FieldResult = Field(default_factory=lambda: FieldResult(unit="D"))
    corneal_astigmatism_axis: FieldResult = Field(default_factory=lambda: FieldResult(unit="degree"))
    pupil: FieldResult = Field(default_factory=lambda: FieldResult(unit="mm"))
    k1: Keratometry = Field(default_factory=Keratometry)
    k2: Keratometry = Field(default_factory=Keratometry)
    tk1: Keratometry = Field(default_factory=Keratometry)
    tk2: Keratometry = Field(default_factory=Keratometry)

    def scalar(self, name: str) -> FieldResult:
        if name.endswith(("_power", "_axis")) and name.split("_")[0] in {"k1", "k2", "tk1", "tk2"}:
            group, part = name.split("_")
            return getattr(getattr(self, group), part)
        return getattr(self, name)

class RunRecord(Contract):
    profile_id: str
    provider_id: str
    model_id: str = ""
    family: str = ""
    outcome: Literal["ok", "partial", "error", "unavailable"] = "ok"
    observations: list[Observation] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    elapsed_seconds: float = 0
    startup_seconds: float | None = None
    page_seconds: list[float] = Field(default_factory=list)
    peak_rss_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    iol_calculations: list[IOLCalculation] = Field(default_factory=list)

class ExtractionResult(Contract):
    schema_version: str = SCHEMA_VERSION
    source_sha256: str
    workflow: str
    patient_id: FieldResult = Field(default_factory=FieldResult)
    gender: FieldResult = Field(default_factory=FieldResult)
    measurement_date: FieldResult = Field(default_factory=FieldResult)
    eyes: dict[Eye, EyeResult] = Field(default_factory=lambda: {"OD": EyeResult(), "OS": EyeResult()})
    verification_state: Literal["verified", "unverified", "needs_review", "error"] = "unverified"
    completeness: dict[str, int] = Field(default_factory=dict)
    runs: list[RunRecord] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    iol_calculations: list[IOLCalculation] = Field(default_factory=list)

    def scalar(self, field: str, eye: Eye | None = None) -> FieldResult:
        return self.eyes[eye].scalar(field) if eye else getattr(self, field)
